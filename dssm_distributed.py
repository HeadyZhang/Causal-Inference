# -*- coding: utf-8 -*-
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from utils.tf_utils import *
import tensorflow as tf

# from keras
_EPSILON = 10e-8


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)


def _binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    # Returns
        A tensor.
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        _epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=output)


def _get_activation(name):
    if name == 'sigmoid':
        return tf.nn.sigmoid
    elif name == 'relu':
        return tf.nn.relu
    elif name == 'tanh':
        return tf.nn.tanh
    else:
        raise ValueError('invalid activation: %s' % name)


class DSSM:
    def __init__(self, x_dim, sparse_x_len, split_index, layer_units=None, layer_activations=None, lr=0.001,
                 batch_norm=False, bn_decay=0.9, alpha=0, beta=0, use_sample_weight=True, logger=None):
        """

        :param x_dim: 输入向量的维度, 注意此维度不包括用来padding的维度
        :param sparse_x_len:
        :param split_index
        :param layer_units:
        :param layer_activations:
        :param lr:
        :param batch_norm:
        :param bn_decay:
        :param input_attn:
        :param multi_hot_dict: {field_index: field_size}
        :param task_type: 'regression' or 'classification'
        :param logger
        """
        self.x_dim = x_dim
        self.sparse_x_len = sparse_x_len
        self.split_index = split_index
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.lr = lr
        self.x_raw = None
        self.inf_pred = None
        self.train_phase = None
        self.x = None
        self.y = None
        self.w = None
        self.pred = None
        self.loss = None
        self.global_step = None
        self.train_op = None
        self.batch_norm = batch_norm
        self.batch_norm_decay = bn_decay
        self.alpha = alpha
        self.beta = beta
        self.use_sample_weight = use_sample_weight
        self.dense_output = None
        self.user_vector = None
        self.item_vector = None
        self.logger = logger

    def train_bn_layer(self, x, scope_bn):
        return batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, is_training=True, reuse=None,
                          trainable=True, scope=scope_bn)

    def inf_bn_layer(self, x, scope_bn):
        return batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, is_training=False, reuse=None,
                          trainable=False, scope=scope_bn)

    def forward(self, x, is_train):
        # 定义batch_norm layer
        if self.batch_norm:
            if is_train:
                self.batch_norm_layer = self.train_bn_layer
            else:
                self.batch_norm_layer = self.inf_bn_layer

        # 由于输入是稀疏表示multi-hot向量, 因此第一层NN的参数矩阵乘输入(Wx)可以通过embedding_lookup后reduce_sum计算
        mat = tf.get_variable(name='layer0_kernel', shape=(self.x_dim, self.layer_units[0]),
                              dtype=tf.float32, initializer=tf.initializers.random_normal(stddev=0.01))
        user_bias = tf.get_variable(name='user_layer0_bias', shape=(self.layer_units[0],),
                                    dtype=tf.float32, initializer=tf.initializers.zeros())
        item_bias = tf.get_variable(name='item_layer0_bias', shape=(self.layer_units[0],),
                                    dtype=tf.float32, initializer=tf.initializers.zeros())

        # user tower layer0
        user_x = tf.nn.embedding_lookup(mat, x[:, :self.split_index])   # shape=(None, split_index, layer_units[0])
        user_x = tf.reduce_sum(user_x, axis=1)      # shape=(None, layer_units[0])
        user_x = tf.nn.relu(user_x + user_bias)     # shape=(None, layer_units[0])

        # item tower layer0
        item_x = tf.nn.embedding_lookup(mat, x[:, self.split_index:])   # shape=(None, sparse_x_len - split_index, layer_units[0])
        item_x = tf.reduce_sum(item_x, axis=1)      # shape=(None, layer_units[0])
        item_x = tf.nn.relu(item_x + item_bias)     # shape=(None, layer_units[0])

        with tf.variable_scope('user_nn', reuse=tf.AUTO_REUSE):
            # user tower
            dense_layer = user_x
            for i, (units, activation) in enumerate(zip(self.layer_units[1: -1], self.layer_activations[1: -1])):
                dense_layer = tf.layers.Dense(units=units,
                                              activation=_get_activation(activation),
                                              name='user_layer_%d' % (i + 1),
                                              kernel_initializer=tf.initializers.random_normal(stddev=0.01))(dense_layer)
                if self.batch_norm:
                    # 进行bn
                    dense_layer = self.batch_norm_layer(dense_layer, scope_bn="bn_%d" % i)

            # dense_output shape=[None, output_unit], 这里output_unit=1
            user_vector = tf.layers.Dense(units=self.layer_units[-1],
                                          activation=_get_activation(self.layer_activations[-1]),
                                          name='user_vector',
                                          kernel_initializer=tf.initializers.random_normal(stddev=0.01))(dense_layer)

        with tf.variable_scope('item_nn', reuse=tf.AUTO_REUSE):
            # item tower
            dense_layer = item_x
            for i, (units, activation) in enumerate(zip(self.layer_units[1: -1], self.layer_activations[1: -1])):
                dense_layer = tf.layers.Dense(units=units,
                                              activation=_get_activation(activation),
                                              name='item_layer_%d' % (i + 1),
                                              kernel_initializer=tf.initializers.random_normal(stddev=0.01))(dense_layer)
                if self.batch_norm:
                    # 进行bn
                    dense_layer = self.batch_norm_layer(dense_layer, scope_bn="bn_%d" % i)

            # dense_output shape=[None, output_unit], 这里output_unit=1
            item_vector = tf.layers.Dense(units=self.layer_units[-1],
                                          activation=_get_activation(self.layer_activations[-1]),
                                          name='item_vector',
                                          kernel_initializer=tf.initializers.random_normal(stddev=0.01))(dense_layer)

        # user-item向量内积
        inner_product = tf.reduce_sum(user_vector * item_vector, axis=1, keepdims=False)
        pred = tf.sigmoid(inner_product)
        return pred, user_vector, item_vector

    def cal_loss(self, pred, y, w=None):
        # cross entropy
        origin_loss = _binary_crossentropy(tf.cast(y, dtype=tf.float32), pred)
        sample_loss = origin_loss * (w if w is not None and self.use_sample_weight else 1)
        sample_loss = tf.reduce_mean(sample_loss)

        # 所有参数组成的向量，用以添加正则项
        var_list = tf.trainable_variables(scope='model')
        _var_vector = tf.concat([tf.reshape(v, shape=(-1,)) for v in var_list], axis=0)
        self.logger.info('参数量: %s' % _var_vector.shape)
        self.param_num = _var_vector.shape
        reg_loss = self.alpha * (self.beta * tf.reduce_sum(tf.abs(_var_vector))
                                 + (1 - self.beta) * tf.reduce_sum(tf.square(_var_vector)))
        loss = sample_loss + reg_loss
        return loss, tf.reduce_mean(origin_loss), reg_loss

    def build_model(self, x=None, y=None, w=None, mode=None, hooks=None):
        """
        支持直接传入Tensor, 否则新建placeholder
        :param x
        :param y
        :param w
        :param mode tf.estimator.ModeKeys, 指定train, eval, predict
        :param hooks
        :return:
        """
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            if x is not None:
                self.x = x
            else:
                self.x = tf.placeholder(tf.int32, shape=[None, self.sparse_x_len], name='x')
            if y is not None:
                self.y = y
            else:
                self.y = tf.placeholder(tf.int32, shape=[None, 1], name='y')
            if w is not None:
                self.w = w
            else:
                self.w = None
            # else:
            #     self.w = tf.placeholder(tf.float32, shape=[None, 1], name='w')

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            if mode == tf.estimator.ModeKeys.TRAIN:
                # 训练时无需关注user向量和item向量
                self.pred, _, _ = self.forward(self.x, is_train=True)
            elif mode == tf.estimator.ModeKeys.EVAL:
                # 评估时无需关注user向量和item向量, 但is_train为False
                self.pred, _, _ = self.forward(self.x, is_train=False)
            else:
                # 推断时需要关注user向量和item向量, 且is_train为False
                self.pred, self.user_vector, self.item_vector = self.forward(self.x, is_train=False)

            if mode == tf.estimator.ModeKeys.PREDICT:
                # 对于推理过程, 不应计算loss, 否则会强制要求输入y
                pass
            else:
                self.loss, self.origin_loss, self.reg_loss = self.cal_loss(self.pred, self.y, self.w)
                self.summary = tf.summary.scalar('loss', self.loss)

            # 注意输出所有训练参数, 确保无误
            self.logger.info(tf.trainable_variables())

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
                if mode == tf.estimator.ModeKeys.TRAIN:
                    # 训练过程
                    self.train_op = tf.train.AdamOptimizer(self.lr) \
                        .minimize(self.loss, global_step=self.global_step)
                    spec = tf.estimator.EstimatorSpec(
                        mode=mode,
                        loss=self.loss,
                        predictions=self.pred,
                        train_op=self.train_op,
                        training_hooks=hooks
                    )
                    tf.summary.scalar('loss', self.loss)
                elif mode == tf.estimator.ModeKeys.PREDICT:
                    # 推理过程
                    out_dict = {
                        'output_pred': self.pred,
                        'output_user_vector': self.user_vector,
                        'output_item_vector': self.item_vector
                    }
                    export_outputs = {
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            tf.estimator.export.PredictOutput(out_dict)}
                    spec = tf.estimator.EstimatorSpec(mode=mode,
                                                      predictions=out_dict,
                                                      export_outputs=export_outputs)
                elif mode == tf.estimator.ModeKeys.EVAL:
                    # 评估过程
                    eval_metric_ops = {
                        'sample_loss': tf.metrics.mean(self.origin_loss),
                        'reg_loss': tf.metrics.mean(self.reg_loss),
                        'auc': tf.metrics.auc(self.y, self.pred)
                    }

                    spec = tf.estimator.EstimatorSpec(mode=mode,
                                                      predictions=self.pred,
                                                      loss=self.loss,
                                                      eval_metric_ops=eval_metric_ops)

                else:
                    raise ValueError('invalid mode: %s' % mode)
                return spec
