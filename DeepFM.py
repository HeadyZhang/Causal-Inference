# -*- coding: utf-8 -*-
import tensorflow as tf
from utils.tf_utils import *
from functools import reduce


class DeepFMClassifier:

    def __init__(self, x_dim, sparse_x_len, emb_size, layer_units=None, layer_activations=None, lr=0.001,
                 batch_norm=False, bn_decay=0.9, alpha=1.8, beta=0.5, multi_hot_dict=None,
                 nn_init='glorot_uniform', distributed_type=None, logger=None, use_sample_weight=False,
                 fl_gamma=0., fl_alpha=1., task_type='classification', regression_label_scale=1., round_label=True,
                 iter_step=100):
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
        self.every_n_iter = iter_step


    def build_mlp_network(self, input_tensor, layer_units, layer_activations, is_train, disable_last_bn=False):
        input_x = input_tensor
        for i, (units, activation) in enumerate(zip(layer_units, layer_activations)):
            if activation == 'None':
                activation = None
            input_x = tf.keras.layers.Dense(units, activation=activation)(input_x)
            if self.batch_norm:
                if disable_last_bn and i == len(layer_units) - 1:
                    continue
                input_x = tf.keras.layers.BatchNormalization(trainable=is_train)(input_x)

        return input_x

    def focal_loss(self, y, p, gamma=0., alpha=1.):
        """
        基本思想: 在CE的基础上乘上一个调制因子factor, 对于CE LOSS更高的样本, 其factor更大. (即增加偏离label较大的样本的权重)
        :param y: label tensor, 只能取0或1, shape=(None,)
        :param p: predition tensor, shape=(None,)
        :param gamma: focusing parameter, 大于等于0, 取0时相当于不进行focus
        :param alpha: 用于正负样本均衡, 暂不实现
        :return:
        """
        _epsilon = 1e-7
        p = tf.clip_by_value(p, _epsilon, 1 - _epsilon)  # 限制概率区间, 防止后续出现log(0)的情况
        pt = y * p + (1 - y) * (1 - p)  # pt = p if y == 1 else 1 - p
        ce = -tf.compat.v1.log(pt)  # cross entropy
        factor = tf.pow(1 - pt, gamma)  # 调制因子, 由于0 < 1-pt < 1, 故当pt为常量时(1-pt)^gamma为递减函数, 即gamma越靠近0, 不确定样本的权重越高(越接近1)
        return ce * factor

    def forward(self, x, is_train):
        """

        :param x: shape=(None, sparse_x_len)
        :param is_train:
        :return:
        """
        # 定义batch_norm layer


        # fm前向过程
        # factorization machine只支持二分类
        with tf.compat.v1.variable_scope('var/fm_first_order', reuse=tf.compat.v1.AUTO_REUSE):
            self.ordr1_weights = tf.compat.v1.get_variable('weight_matrix',
                                                 shape=[self.x_dim, 1],
                                                 dtype=tf.float32,
                                                 initializer=tf.initializers.random_normal(stddev=0.01))
            # # 使用最大的index来作为空值
            # ordr1_weights = tf.concat([self.ordr1_weights, tf.zeros(shape=[1, 1])], axis=0)  # [x_dim + 1, 1]
            # ^^其实没必要, 当lookup的id大于lookup param维度时, tf会直接返回0

            # 查找每个离散值index对应维度的order1权重
            first_ordr = tf.nn.embedding_lookup(self.ordr1_weights, x)  # [None, sparse_x_len, 1]
            first_ordr = tf.reduce_sum(first_ordr, 2)  # [None, sparse_x_len], deepFM中用于拼接的一阶向量
            intersect = tf.compat.v1.get_variable('intersect', shape=[1], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
            fm_first_ordr = tf.reduce_sum(first_ordr, axis=1, keepdims=True)

            # intersect重复batch_size次后进行拼接
            pred = fm_first_ordr + intersect
        with tf.compat.v1.variable_scope('var/fm_second_order', reuse=tf.compat.v1.AUTO_REUSE):
            self.ordr2_emb_mat = tf.compat.v1.get_variable('embedding_matrix',
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
            with tf.compat.v1.variable_scope('embedding_layer', reuse=tf.compat.v1.AUTO_REUSE):
                # 使用fm的embedding作为nn的输入层
                emb_layer = tf.reshape(self.dis_embs,  shape=(-1, self.sparse_x_len * self.emb_size))

                # self.emb_layer = tf.reduce_mean(self.dis_embs, axis=1)  # [None, emb_size]

            with tf.compat.v1.variable_scope('var/dense_nn', reuse=tf.compat.v1.AUTO_REUSE):
                # 最底层nn, 由于参数量最大, 因此主label网络和辅助label网络共用
                logits = self.build_mlp_network(
                    input_tensor=emb_layer,
                    layer_units=self.layer_units,
                    layer_activations=self.layer_activations,
                    is_train=is_train,
                    disable_last_bn=False)
                pred += logits
        pred = tf.sigmoid(pred)
        return pred

    def cal_loss(self):
        # shape=[None,]
        # if self.task_type == 'classification':
        # use focal loss
        sample_loss = self.focal_loss(
                self.y,
                self.pred,
                gamma=self.fl_gamma,
                alpha=self.fl_alpha)

        if self.w is not None:
            sample_loss *= self.w                            # shape=[None,]
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

        with tf.compat.v1.variable_scope('input', reuse=tf.compat.v1.AUTO_REUSE):
            # self.x_raw = tf.placeholder(tf.int32, shape=[None, self.sparse_x_len], name='x_raw')   # 用于提供服务的输入
            # self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            self.x = tf.reshape(x, shape=(-1, self.sparse_x_len))   # 任何情况下x都不能为空

            if y is not None:
                # 对于推理过程, y可以为空
                self.y = tf.cast(tf.reshape(y, [-1]), dtype=tf.float32)

        with tf.compat.v1.variable_scope('train', reuse=tf.compat.v1.AUTO_REUSE):
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.pred = self.forward(self.x, is_train=True)
            else:
                self.pred = self.forward(self.x, is_train=False)
            self.pred = tf.reshape(self.pred, [-1])

            if mode == tf.estimator.ModeKeys.PREDICT:
                # 对于推理过程, 不应计算loss, 否则会强制要求输入y
                # 推理过程
                out_dict = {'prediction': self.pred}
                export_outputs = {
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(out_dict)
                }
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  predictions=out_dict,
                                                  export_outputs=export_outputs)

            # 非推理过程, 需要计算loss
            self.w = w if self.use_sample_weight else None
            self.loss  = self.cal_loss()
            tf.compat.v1.summary.scalar('loss', self.loss)
            eval_metric_ops = {

            }

            tensors_to_log = {
                "loss": self.loss
            }
            logging_hook = tf.estimator.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=self.every_n_iter)
            hooks = [logging_hook, tf.estimator.StepCounterHook()]

            eval_metric_ops['auc'] = tf.compat.v1.metrics.auc(tf.cast(self.y, tf.int64), self.pred)
            eval_metric_ops['recall'] = tf.compat.v1.metrics.recall(tf.cast(self.y, tf.int64), tf.greater_equal(self.pred, tf.constant(0.5)))
            eval_metric_ops['precision'] = tf.compat.v1.metrics.precision(tf.cast(self.y, tf.int64), tf.greater_equal(self.pred, tf.constant(0.5)))

            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.lr)
                grads, variables = zip(*optimizer.compute_gradients(self.loss))
                grads, global_norm = tf.clip_by_global_norm(grads, 5)
                self.train_op = optimizer.apply_gradients(zip(grads, variables),
                                                          global_step=tf.compat.v1.train.get_or_create_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, train_op=self.train_op,
                                                  training_hooks=hooks)

            elif mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, eval_metric_ops=eval_metric_ops,
                                                  evaluation_hooks=hooks)
            else:
                raise ValueError('invalid mode: %s' % mode)



if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()

    layer_units = [32,32,1]
    layer_activitions = ['relu', 'relu', 'sigmoid']
    batch_size, x_dim, sparse_x_len, emb_size = 3, 100, 20, 5
    data_x = np.random.random_integers(0, x_dim - 1, size=(batch_size, sparse_x_len))

    data_y = [[1],
              [1],
              [0]]

    data_w = np.random.random_integers(1, 5, size=(batch_size, 1))

    placeholder_x = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None, sparse_x_len))
    placeholder_y = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None, 1))
    placeholder_w = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 1))

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

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        for _ in range(100):
            _, loss, pred = sess.run([model.train_op, model.loss, model.pred],
                                       feed_dict={model.x: data_x,
                                                  model.y: data_y,
                                                  model.w: data_w
                                                  })
            print(loss)
        print(pred)





