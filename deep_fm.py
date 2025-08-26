# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from utils.tf_utils import *


class DeepFMClassifier:

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

        self.w = None
        self.sample_loss = None
        self.reg_loss = None
        self.param_num = None
        self.summary = None

        self.batch_norm_layer = None

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

        # fm前向过程
        # factorization machine只支持二分类
        with tf.variable_scope('var/fm_first_order', reuse=tf.AUTO_REUSE):
            self.ordr1_weights = tf.get_variable('weight_matrix',
                                                 shape=[self.x_dim, 1],
                                                 dtype=tf.float32,
                                                 initializer=tf.initializers.random_normal(stddev=0.01))
            # # 使用最大的index来作为空值
            # ordr1_weights = tf.concat([self.ordr1_weights, tf.zeros(shape=[1, 1])], axis=0)  # [x_dim + 1, 1]
            # ^^其实没必要, 当lookup的id大于lookup param维度时, tf会直接返回0

            # 查找每个离散值index对应维度的order1权重
            first_ordr = tf.nn.embedding_lookup(self.ordr1_weights, x)  # [None, sparse_x_len, 1]
            first_ordr = tf.reduce_sum(first_ordr, 2)  # [None, sparse_x_len], deepFM中用于拼接的一阶向量
            intersect = tf.get_variable('intersect', shape=[1], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
            fm_first_ordr = tf.reduce_sum(first_ordr, axis=1, keepdims=True)
            pred = fm_first_ordr + intersect

        with tf.variable_scope('var/fm_second_order', reuse=tf.AUTO_REUSE):
            self.ordr2_emb_mat = tf.get_variable('embedding_matrix',
                                                shape=[self.x_dim, self.emb_size],
                                                     dtype=tf.float32,
                                                     initializer=tf.initializers.random_normal(stddev=0.01))
            # 使用最大的index来作为空值
            # ordr2_emb_mat = tf.concat([self.ordr2_emb_mat, tf.zeros(shape=[1, self.emb_size])],
            #                           axis=0)  # [x_dim + 1, emb_size]
            self.dis_embs = tf.nn.embedding_lookup(self.ordr2_emb_mat, x)  # [None, sparse_x_len, emb_size]
            # sum -> sqr part
            sum_dis_embs = tf.reduce_sum(self.dis_embs, 1)  # [None, emb_size]
            sum_sqr_dis_embs = tf.square(sum_dis_embs)  # [None, emb_size]
            # sqr -> sum part
            sqr_dis_embs = tf.square(self.dis_embs)  # [None, sparse_x_len, emb_size]
            sqr_sum_dis_embs = tf.reduce_sum(sqr_dis_embs, 1)  # [None, emb_size]
            # second order
            second_ordr = 0.5 * (sum_sqr_dis_embs - sqr_sum_dis_embs)  # [None, emb_size], deepFM中用于拼接的二阶向量
            fm_second_ordr = tf.reduce_sum(second_ordr, axis=1, keepdims=True)
            pred += fm_second_ordr


        if self.layer_units is not None and self.layer_activations is not None:
            # 神经网络参数非空, 进行deepfm
            with tf.variable_scope('embedding_layer', reuse=tf.AUTO_REUSE):
                # 使用fm的embedding作为nn的输入层
                emb_layer = tf.reshape(self.dis_embs,  shape=(-1, self.sparse_x_len * self.emb_size))

                # self.emb_layer = tf.reduce_mean(self.dis_embs, axis=1)  # [None, emb_size]

            with tf.variable_scope('var/dense_nn', reuse=tf.AUTO_REUSE):
                # 最底层nn, 由于参数量最大, 因此主label网络和辅助label网络共用
                logits = get_mlp_network(
                    input_tensor=emb_layer,
                    layer_units=self.layer_units,
                    layer_activations=self.layer_activations,
                    name='nn_network',
                    bn_layer=self.batch_norm_layer,
                    disable_last_bn=True
                )
                pred += logits
        pred = tf.sigmoid(pred)
        return pred

    def cal_loss(self, pred, y, w=None):
        # shape=[None,]
        # if self.task_type == 'classification':
        # use focal loss
        sample_loss = focal_loss(
                tf.cast(tf.reshape(y, [-1]), dtype=tf.float32),
                tf.reshape(pred, [-1]),
                gamma=self.fl_gamma,
                alpha=self.fl_alpha)

        if w is not None:
            sample_loss *= w                            # shape=[None,]
        sample_loss = tf.reduce_mean(sample_loss)       # shape=[] (scalar)

        # 所有参数组成的向量，用以添加正则项
        # var_list = tf.trainable_variables(scope='train/var')
        # _var_vector = tf.concat([tf.reshape(v, shape=(-1,)) for v in var_list], axis=0)
        # if self.mode == tf.estimator.ModeKeys.TRAIN:
        #     for v in var_list:
        #         self.logger.info('reg var: %s' % v)
        #         self.logger.info('*' * 50)
        #     self.logger.info('参数量: %s' % _var_vector.shape)
        # self.param_num = _var_vector.shape
        # reg_loss = self.alpha * (self.beta * tf.reduce_sum(tf.abs(_var_vector))
        #                          + (1 - self.beta) * tf.reduce_sum(tf.square(_var_vector)))
        # loss = sample_loss + reg_loss
        return  sample_loss

    def build_model(self, x=None, y=None, w=None, mode=None, hooks=None):
        self.mode = mode
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
            # self.x_raw = tf.placeholder(tf.int32, shape=[None, self.sparse_x_len], name='x_raw')   # 用于提供服务的输入
            # self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            self.x = tf.reshape(x, shape=(-1, self.sparse_x_len))   # 任何情况下x都不能为空

            if y is not None:
                # 对于推理过程, y可以为空
                self.y = tf.cast(y, dtype=tf.float32)

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
                self.w = w if self.use_sample_weight else None
                self.loss  = self.cal_loss(self.pred, self.y, w=self.w)
                self.summary = tf.summary.scalar('loss', self.loss)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if mode == tf.estimator.ModeKeys.TRAIN:
                    # 训练过程
                    self.train_op = tf.train.AdamOptimizer(self.lr) \
                            .minimize(self.loss, global_step=self.global_step)
                    spec = tf.estimator.EstimatorSpec(
                            mode=mode,
                            loss=self.loss,
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
                            # 'sample_loss': tf.metrics.mean(self.sample_loss),
                            # 'reg_loss': tf.metrics.mean(self.reg_loss)
                    }

                    eval_metric_ops['auc'] = tf.metrics.auc(tf.cast(self.y, tf.int64), self.pred)
                    eval_metric_ops['recall'] = tf.metrics.recall(tf.cast(self.y, tf.int64), tf.greater_equal(self.pred, 0.5))
                    eval_metric_ops['precision'] = tf.metrics.precision(tf.cast(self.y, tf.int64), tf.greater_equal(self.pred, 0.5))
                    spec = tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=self.loss,
                                                      eval_metric_ops=eval_metric_ops)

                else:
                    raise ValueError('invalid mode: %s' % mode)
                return spec



if __name__ == '__main__':
    layer_units = '32,32,1'
    layer_activitions = 'relu,relu,sigmoid'
    batch_size, x_dim, sparse_x_len, emb_size = 3, 100, 20, 5
    data_x = np.random.random_integers(0, x_dim - 1, size=(batch_size, sparse_x_len))

    data_y = [[1],
              [1],
              [0]]

    data_w = np.random.random_integers(1, 5, size=(batch_size, 1))

    placeholder_x = tf.placeholder(dtype=tf.int64, shape=(None, sparse_x_len))
    placeholder_y = tf.placeholder(dtype=tf.int64, shape=(None, 1))
    placeholder_w = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    model = DeepFMClassifier(x_dim,
                      sparse_x_len,
                      emb_size,
                      layer_units=layer_units,
                      layer_activations=layer_activitions,
                      fl_gamma=0.0,
                      fl_alpha=1.0,
                      alpha=0.5,
                      beta=0.2,
                      lr=0.001,
                      logger=None,
                      bn_decay=0.9,
                      use_sample_weight=True,
                      batch_norm=False)

    model.build_model(x=placeholder_x, y=placeholder_y, w=placeholder_w, mode=tf.estimator.ModeKeys.TRAIN)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for _ in range(100):
            _, loss, pred = sess.run([model.train_op, model.loss, model.pred],
                                       feed_dict={model.x: data_x,
                                                  model.y: data_y,
                                                  model.w: data_w
                                                  })
            print(loss)
        print(pred)





