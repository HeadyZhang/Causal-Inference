# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from utils.tf_utils import *


class DNN:

    def __init__(self, x_dim, sparse_x_len, emb_size, layer_units=None, layer_activations=None, lr=0.001,
                 batch_norm=False, bn_decay=0.9, alpha=1.8, beta=0.5, multi_hot_dict=None,
                 nn_init='glorot_uniform', distributed_type=None, logger=None, use_sample_weight=False,
                 fl_gamma=0., fl_alpha=1., task_type='classification', regression_label_scale=1., round_label=True):
        """

        :param x_dim:
        :param sparse_x_len:
        :param emb_size: fm embedding size
        :param layer_units: deepfm nn units
        :param layer_activations: deepfm nn activations
        :param lr: learning rate
        :param batch_norm: 是否进行batch normalization
        :param bn_decay: batch normalization decay参数
        :param alpha: 正则化参数, 总权重
        :param beta: 正则化参数, L1/L2占比
        :param multi_hot_dict: 标明哪些field是multi-hot的 {field_index: field_size}
        :param nn_init: nn参数初始化方法
        :param distributed_type: 分布式训练同步类型
        :param logger:
        :param use_sample_weight: 是否使用sample weight
        :param fl_gamma: focal loss的gamma值
        :param fl_alpha: focal loss的alpha值
        :param task_type: 'classification' or 'regression'
        :param regression_label_scale
        :param round_label
        """
        # 模型参数设置
        self.x_dim = x_dim
        self.sparse_x_len = sparse_x_len
        self.emb_size = emb_size
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        if self.layer_units is not None and self.layer_activations is not None:
            assert len(self.layer_units) == len(self.layer_activations), 'layer units and activations length not match: %s vs %s' % (len(self.layer_units), len(self.layer_activations))
        self.lr = lr
        self.batch_norm = batch_norm
        self.batch_norm_decay = bn_decay
        self.alpha = alpha
        self.beta = beta
        self.multi_hot_dict = multi_hot_dict
        self.nn_init = nn_init
        self.use_sample_weight = use_sample_weight
        self.distributed_type = distributed_type  # None, sync, async
        if self.distributed_type is not None:
            assert self.distributed_type in ('sync', 'async')
        self.mode = None
        # multi-hot在输入x中对应的下标
        i_multi_hot = []
        for key in multi_hot_dict:
            temp = list(range(key, key + multi_hot_dict[key]))
            i_multi_hot.extend(temp)

        # one-hot在输入x中对应的下标
        self.i_one_hot = []
        for i in range(sparse_x_len):
            if i not in i_multi_hot:
                self.i_one_hot.append(i)

        self.indices, self.values = [], []
        self.fl_gamma = fl_gamma
        self.fl_alpha = fl_alpha
        self.task_type = task_type
        self.regression_label_scale = regression_label_scale
        self.round_label = round_label

        # logger
        self.logger = logger

        # 模型重要tensor声明
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
        self.ordr1_weights = None
        self.ordr2_emb_mat = None
        self.emb_layer = None
        self.dense_output = None
        self.fm_first_ordr = None
        self.fm_second_ordr = None
        self.sample_weight = None
        self.sample_loss = None
        self.reg_loss = None
        self.param_num = None
        self.summary = None

    def train_bn_layer(self, x, scope_bn):
        return batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, is_training=True, reuse=None,
                          trainable=True, scope=scope_bn)

    def inf_bn_layer(self, x, scope_bn):
        return batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, is_training=False, reuse=None,
                          trainable=False, scope=scope_bn)

    def forward(self, x, is_train):
        """

        :param x: shape=(None, sparse_x_len)
        :param is_train:
        :return:
        """
        # 定义batch_norm layer
        if self.batch_norm:
            if is_train:
                self.batch_norm_layer = self.train_bn_layer
            else:
                self.batch_norm_layer = self.inf_bn_layer

        with tf.variable_scope('var/dense_nn', reuse=tf.AUTO_REUSE):
            # 最底层nn, 由于输入是稀疏表达向量, 单独写了函数实现
            dense_layer = sparse_input_layer(indices=x,
                                             x_dim=self.x_dim,
                                             units=self.layer_units[0],
                                             activation=self.layer_activations[0])
            if self.batch_norm:
                # 进行bn
                dense_layer = self.batch_norm_layer(dense_layer, scope_bn="bn_%d" % 0)

            for i, (units, activation) in enumerate(zip(self.layer_units[1: -1], self.layer_activations[1: -1])):
                dense_layer = tf.keras.layers.Dense(units=units,
                                                    activation=activation,
                                                    name='dense_layer_%d' % (i + 1),
                                                    kernel_initializer=self.nn_init)(dense_layer)
                if self.batch_norm:
                    # 进行bn
                    dense_layer = self.batch_norm_layer(dense_layer, scope_bn="bn_%d" % (i + 1))

            # dense_output shape=[None, output_unit], 这里output_unit=1
            dense_output = tf.keras.layers.Dense(units=self.layer_units[-1],
                                                 activation=self.layer_activations[-1],
                                                 name='dense_layer_%d' % (len(self.layer_units) - 1),
                                                 kernel_initializer=self.nn_init)(dense_layer)
        return dense_output

    def cal_loss(self, pred, y, w=None):
        # shape=[None,]
        if self.task_type == 'classification':
            # use focal loss
            sample_loss = focal_loss(
                tf.cast(tf.reshape(y, [-1]), dtype=tf.float32),
                tf.reshape(pred, [-1]),
                gamma=self.fl_gamma,
                alpha=self.fl_alpha
            )
        else:
            # use mse
            _prediction = tf.reshape(pred, [-1])
            # 单位从毫秒转成秒后乘上衰减系数
            _label = tf.cast(tf.reshape(y, [-1]), dtype=tf.float32) / 1000
            if self.round_label:
                _label = tf.round(_label)
            _label *= self.regression_label_scale
            sample_loss = (_prediction - _label) ** 2
        if w is not None:
            sample_loss *= w                            # shape=[None,]
        sample_loss = tf.reduce_mean(sample_loss)       # shape=[] (scalar)

        # 所有参数组成的向量，用以添加正则项
        var_list = tf.trainable_variables(scope='train/var')
        _var_vector = tf.concat([tf.reshape(v, shape=(-1,)) for v in var_list], axis=0)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            for v in var_list:
                self.logger.info('reg var: %s' % v)
                self.logger.info('*' * 50)
            self.logger.info('参数量: %s' % _var_vector.shape)
        self.param_num = _var_vector.shape
        reg_loss = self.alpha * (self.beta * tf.reduce_sum(tf.abs(_var_vector))
                                 + (1 - self.beta) * tf.reduce_sum(tf.square(_var_vector)))
        loss = sample_loss + reg_loss
        return loss, sample_loss, reg_loss

    def build_model(self, x=None, y=None, w=None, mode=None, hooks=None):
        self.mode = mode
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
            self.x = tf.reshape(x, shape=(-1, self.sparse_x_len))   # 任何情况下x都不能为空

            if y is not None:
                # 对于推理过程, y可以为空
                self.y = tf.reshape(y, shape=(-1, 1))

        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.pred = self.forward(self.x, is_train=True)
            else:
                self.pred = self.forward(self.x, is_train=False)

            if mode == tf.estimator.ModeKeys.PREDICT:
                # 对于推理过程, 不应计算loss, 否则会强制要求输入y
                pass
            else:
                # 非推理过程, 需要计算loss
                self.sample_weight = w if self.use_sample_weight else None
                self.loss, self.sample_loss, self.reg_loss = self.cal_loss(self.pred, self.y, w=self.sample_weight)
                self.summary = tf.summary.scalar('loss', self.loss)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self.distributed_type == 'async':
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
                        out_dict = {'prediction': self.pred}
                        export_outputs = {
                            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                tf.estimator.export.PredictOutput(out_dict)}
                        spec = tf.estimator.EstimatorSpec(mode=mode,
                                                          predictions=out_dict,
                                                          export_outputs=export_outputs)
                    elif mode == tf.estimator.ModeKeys.EVAL:
                        # 评估过程
                        eval_metric_ops = {
                            'sample_loss': tf.metrics.mean(self.sample_loss),
                            'reg_loss': tf.metrics.mean(self.reg_loss)
                        }
                        if self.task_type == 'classification':
                            eval_metric_ops['auc'] = tf.metrics.auc(self.y, self.pred)
                        spec = tf.estimator.EstimatorSpec(mode=mode,
                                                          predictions=self.pred,
                                                          loss=self.loss,
                                                          eval_metric_ops=eval_metric_ops)

                    else:
                        raise ValueError('invalid mode: %s' % mode)
                    return spec
                elif self.distributed_type == 'sync':
                    pass
                else:
                    pass


if __name__ == '__main__':
    pass
