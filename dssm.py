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
                 batch_norm=False, bn_decay=0.9, alpha=0, beta=0, use_sample_weight=True):
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
        self.val_x = None
        self.val_y = None
        self.pred = None
        self.loss = None
        self.val_pred = None
        self.val_loss = None
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

    def train_bn_layer(self, x, scope_bn):
        return batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, is_training=True, reuse=None,
                          trainable=True, scope=scope_bn)

    def inf_bn_layer(self, x, scope_bn):
        return batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, is_training=False, reuse=True,
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

        with tf.variable_scope('user_nn'):
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

        with tf.variable_scope('item_nn'):
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
        inner_product = tf.reduce_sum(user_vector * item_vector, axis=1, keepdims=True)
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
        print('参数量: %s' % _var_vector.shape)
        self.param_num = _var_vector.shape
        reg_loss = self.alpha * (self.beta * tf.reduce_sum(tf.abs(_var_vector))
                                 + (1 - self.beta) * tf.reduce_sum(tf.square(_var_vector)))
        loss = sample_loss + reg_loss
        return loss, tf.reduce_mean(origin_loss), reg_loss

    def build_model(self, x=None, y=None, w=None, val_x=None, val_y=None, val_w=None):
        """
        支持直接传入Tensor, 否则新建placeholder
        :param x
        :param y
        :param w
        :param val_x
        :param val_y
        :param val_w
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

            if val_x is not None:
                self.val_x = val_x
            else:
                self.val_x = tf.placeholder(tf.int32, shape=[None, self.sparse_x_len], name='val_x')
            self.val_y = val_y
            self.val_w = val_w

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self.pred, _, _ = self.forward(self.x, is_train=True)
            self.val_pred, self.user_vector, self.item_vector = self.forward(self.val_x, is_train=False)
            if self.val_y is not None:
                self.val_loss, self.val_origin_loss, self.val_reg_loss = self.cal_loss(self.val_pred, self.val_y, self.val_w)
            self.loss, self.origin_loss, self.reg_loss = self.cal_loss(self.pred, self.y, self.w)
            self.summary = tf.summary.scalar('loss', self.loss)

            # 注意输出所有训练参数, 确保无误
            print(tf.trainable_variables())
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)


if __name__ == '__main__':
    model_path = 'temp/best_model'
    pb_model_path = 'temp/pb_model'
    data_x = [
        [1, 14, 20, 35, 41, 50, 60, 70, 80, 92],
        [2, 13, 26, 39, 49, 52, 67, 72, 88, 91],
        [6, 18, 23, 36, 42, 51, 66, 72, 82, 97],
        [8, 12, 24, 32, 43, 57, 61, 77, 82, 96],
        [2, 11, 21, 33, 44, 58, 61, 75, 83, 94]
    ]

    data_y = [
        [1],
        [1],
        [0],
        [0],
        [0]
    ]
    dssm = DSSM(x_dim=100, sparse_x_len=10, split_index=4, layer_units=[128, 128, 128, 128],
                layer_activations=['relu', 'relu', 'relu', 'tanh'], batch_norm=True)
    dssm.build_model()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(100):
            _, loss = sess.run([dssm.train_op, dssm.loss], feed_dict={dssm.x: data_x, dssm.y: data_y})
            print(loss)
        print('train finished')
        # pred = sess.run(dssm.pred, feed_dict={dssm.x: data_x})
        pred, val_pred, u_vec, i_vec = sess.run([dssm.pred, dssm.val_pred, dssm.user_vector, dssm.item_vector], feed_dict={dssm.x: data_x, dssm.val_x: data_x})
        print(pred)
        print(val_pred)
        print(u_vec)
        print(i_vec)
        saver = tf.train.Saver()
        saver.save(sess, model_path)

        # 模型导出
        save_pb_model(
            model_out_path=pb_model_path,
            input_dict={'input_x': dssm.val_x},
            output_dict={'output_pred': dssm.val_pred, 'output_user_vector': dssm.user_vector, 'output_item_vector': dssm.item_vector},
            session=sess,
            mode='overwrite'
        )

    with tf.Session() as sess:
        # preparing tf model
        signature = tf.saved_model.loader.load(sess,
                                               tags=[tf.saved_model.tag_constants.SERVING],
                                               export_dir=pb_model_path) \
            .signature_def
        x_name = signature[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] \
            .inputs['input_x'].name
        pred_name = signature[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] \
            .outputs['output_pred'].name
        user_vector_name = signature[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] \
            .outputs['output_user_vector'].name
        item_vector_name = signature[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] \
            .outputs['output_item_vector'].name

        x = sess.graph.get_tensor_by_name(x_name)
        pred = sess.graph.get_tensor_by_name(pred_name)
        user_vector = sess.graph.get_tensor_by_name(user_vector_name)
        item_vector = sess.graph.get_tensor_by_name(item_vector_name)

        pred, u_vec, i_vec = sess.run([pred, user_vector, item_vector], feed_dict={x: data_x})

        print(pred)
        print(u_vec)
        print(i_vec)

    # x_name = 'input/x:0'
    # val_x_name = 'input/val_x:0'
    # pred_name = 'model/Sigmoid:0'
    # val_pred_name = 'model/Sigmoid_1:0'
    # user_vector_name = 'model/user_nn_1/user_vector/Tanh:0'
    # item_vector_name = 'model/item_nn_1/item_vector/Tanh:0'
    #
    # with tf.Session() as sess:
    #     print('*********************************************************')
    #     saver = tf.train.Saver()
    #     saver.restore(sess, model_path)
    #     # assign tensor
    #     val_pred = tf.get_default_graph().get_tensor_by_name(val_pred_name)
    #     user_vector = tf.get_default_graph().get_tensor_by_name(user_vector_name)
    #     item_vector = tf.get_default_graph().get_tensor_by_name(item_vector_name)
    #
    #     val_pred, u_vec, i_vec = sess.run([val_pred, user_vector, item_vector], feed_dict={dssm.val_x: [data_x[0]]})
    #     print(val_pred[0])
    #     print(u_vec[0])
    #     print(i_vec[0])
