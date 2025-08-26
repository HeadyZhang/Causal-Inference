# -*- coding: utf-8 -*-
import tensorflow as tf
from utils.tf_utils import *
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm



class Dragonnet:
    def __init__(self, x_dim, sparse_x_len, emb_size, layer_units=None, layer_activations=None,
                 tower_units=None, tower_activations=None, lr=0.001, g_weight=None,
                 batch_norm=False, bn_decay=0.9, alpha=1.0, beta=0.5,
                 nn_init='glorot_uniform', logger=None, use_sample_weight=False,
                 fl_gamma=0., fl_alpha=1.,  enable_drm=False, drm_param=0, g_propensity=False):
        # logger
        self.logger = logger

        # 模型参数设置
        self.x_dim = x_dim
        self.sparse_x_len = sparse_x_len
        self.emb_size = emb_size
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.tower_units = tower_units
        self.tower_activations = tower_activations
        self.lr = lr
        self.g_weight = g_weight
        self.batch_norm = batch_norm
        self.batch_norm_decay = bn_decay
        self.alpha = alpha
        self.beta = beta
        self.nn_init = nn_init
        self.use_sample_weight = use_sample_weight
        self.mode = None
        # multi-hot在输入x中对应的下标

        # one-hot在输入x中对应的下标
        self.i_one_hot = []


        self.indices, self.values = [], []
        self.fl_gamma = fl_gamma
        self.fl_alpha = fl_alpha
        self.enable_drm = enable_drm
        self.drm_param = drm_param

        # 模型重要tensor声明
        self.x = None
        self.masked_x = None
        self.y = None
        self.y_outcome = None
        self.y_treatment = None
        self.w = None
        self.q0 = None
        self.q1 = None
        self.g = None
        self.l0 = None
        self.l1 = None
        self.lg = None

        self.loss = None
        self.val_loss = None
        self.global_step = None
        self.train_op = None
        self.summary = None
        self.batch_norm_layer = None
        self.epsilon = None

        self.g_propensity = g_propensity

    def train_bn_layer(self, x, scope_bn):
        return batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, is_training=True, reuse=None,
                          trainable=True, scope=scope_bn)

    def inf_bn_layer(self, x, scope_bn):
        return batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, is_training=False, reuse=None,
                          trainable=False, scope=scope_bn)

    def forward(self, is_train):
        """
        :param is_train:
        :return:
        """
        # 定义batch_norm layer
        if self.batch_norm:
            if is_train:
                self.batch_norm_layer = self.train_bn_layer
            else:
                self.batch_norm_layer = self.inf_bn_layer

        emb_mat = tf.get_variable('embedding_matrix',
                                  shape=[self.x_dim, self.emb_size],
                                  dtype=tf.float32,
                                  initializer=tf.initializers.random_normal(stddev=0.01))

        dis_embs = tf.nn.embedding_lookup(emb_mat, self.masked_x)

        dense_x = tf.reshape(dis_embs, shape=(-1, self.sparse_x_len * self.emb_size))

        z = get_mlp_network(dense_x, self.layer_units, self.layer_activations, 'base_network', bn_layer=self.batch_norm_layer)

        q0 = get_mlp_network(z, self.tower_units, self.tower_activations, 'no_treatment_network', bn_layer=self.batch_norm_layer, disable_last_bn=True)
        q1 = get_mlp_network(z, self.tower_units, self.tower_activations, 'treatment_network', bn_layer=self.batch_norm_layer, disable_last_bn=True)
        g = get_mlp_network(z, [1], ['sigmoid'], 'g_network')

        return q0, q1, g

    def cal_loss(self):
        # 所有参数组成的向量，用以添加正则项
        self.y_outcome, self.y_treatment = tf.reshape(self.y[:, 0], (-1, 1)), tf.reshape(self.y[:, 1], (-1, 1))
        if self.g_propensity:
            self.l0 = focal_loss(self.q0, self.y_outcome) * (1 - self.y_treatment) /(1 + tf.exp(1 - self.g))
            self.l1 = focal_loss(self.q1, self.y_outcome) * self.y_treatment /(1+ tf.exp(self.g))
        else:
            self.l0 = focal_loss(self.q0, self.y_outcome) * (1 - self.y_treatment)
            self.l1 = focal_loss(self.q1, self.y_outcome) * self.y_treatment

        self.lg = focal_loss(self.g, self.y_treatment) * self.g_weight

        if self.use_sample_weight:
            w_outcome, w_treatment = tf.reshape(self.w[:, 0], (-1, 1)), tf.reshape(self.w[:, 1], (-1, 1))
        else:
            w_outcome, w_treatment = 1, 1

        self.loss = tf.reduce_mean(self.l0 * w_outcome  + self.l1 * w_outcome + self.lg)

    def build_model(self, x=None, y=None, w=None, mode=None, hooks=None):
        """

        :param x: tensor类型, 稀疏表达的multi-hot向量, 包含所有向量值为1的indices
        :param x_seq: tensor类型, 序列输入
        :param y: tensor类型, label. 其中y[0]对应outcome, y[1]对应是否treatment
        :param w: tensor类型, sample weight. 其中w[0]对应outcome, w[1]对应是否treatment
        :param mode: tf.estimator.ModeKeys的某个枚举值, 用于指定模型处于训练、评估或是推断模式
        :param hooks: TensorFlow hook
        :return:
        """
        self.mode = mode
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
            self.x = tf.reshape(x, shape=(-1, self.sparse_x_len))   # 任何情况下x都不能为空

            # mask: 去除掉不用的特征

            self.masked_x = self.x

            if y is not None:
                # 对于推理过程, y可以为空
                self.y = tf.cast(tf.reshape(y, (-1, 2)), dtype=tf.float32)

        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.q0, self.q1, self.g = self.forward(is_train=True)     # shape=[None, n_towers]
            else:
                self.q0, self.q1, self.g = self.forward(is_train=False)    # shape=[None, n_towers]

            if mode == tf.estimator.ModeKeys.PREDICT:
                # 对于推理过程, 不应计算loss, 否则会强制要求输入y
                pass
            else:
                # 非推理过程, 需要计算loss
                self.w = w
                self.cal_loss()
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
                    out_dict = {'prediction': tf.concat([self.q0, self.q1, self.g], axis=1)}           # shape=[None, 3]

                    export_outputs = {
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            tf.estimator.export.PredictOutput(out_dict)}
                    spec = tf.estimator.EstimatorSpec(mode=mode,
                                                      predictions=out_dict,
                                                      export_outputs=export_outputs)
                elif mode == tf.estimator.ModeKeys.EVAL:
                    # 评估过程
                    eval_metric_ops = {
                        'g_auc': tf.metrics.auc(self.y_treatment, self.g),
                        'tr_auc': tf.metrics.auc(self.y_outcome, self.q1),
                        'cr_auc': tf.metrics.auc(self.y_outcome, self.q0),
                        'loss_0': tf.metrics.mean(self.l0),
                        'loss_1': tf.metrics.mean(self.l1),
                        'loss_g': tf.metrics.mean(self.lg)
                    }
                    if self.enable_drm:
                        eval_metric_ops.update({
                            'drm_loss': tf.metrics.mean(self.drm_loss)
                        })

                    spec = tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=self.loss,
                                                      eval_metric_ops=eval_metric_ops)
                else:
                    raise ValueError('invalid mode: %s' % mode)
                return spec

    def print(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)


if __name__ == '__main__':
    # 可直接运行, 测试类的逻辑正确性
    units = [64, 64, 64]
    activations = ['relu', 'relu', 'sigmoid']
    tower_units = [16, 16, 1]
    tower_activations = ['relu', 'relu', 'sigmoid']
    seq_dim = 11
    context_dim = 8
    sparse_context_range = [1, 9]
    position_attention = False
    enable_drm = True

    # mock data
    batch_size, x_dim, sparse_x_len, emb_size = 32, 100, 20, 5
    data_x = np.random.random_integers(0, x_dim - 1, size=(batch_size, sparse_x_len))
    # data_y = np.random.random_integers(0, 1, size=(batch_size, 2))

    # data_y = [[1, 0],
    #           [1, 1],
    #           [0, 1]]
    data_y = np.random.randint(0, 2, size=(batch_size, 2))
    data_w = np.random.random_integers(1, 5, size=(batch_size, 2))

    # place holders
    placeholder_x = tf.placeholder(dtype=tf.int64, shape=(None, sparse_x_len))
    placeholder_y = tf.placeholder(dtype=tf.int64, shape=(None, 2))
    placeholder_w = tf.placeholder(dtype=tf.float32, shape=(None, 2))

    # model init
    model = Dragonnet(x_dim, sparse_x_len, emb_size, layer_units=units, layer_activations=activations,
                      tower_units=tower_units, tower_activations=tower_activations, lr=0.001, g_weight=1.,
                      batch_norm=False, bn_decay=0.9, alpha=1.8, beta=0.5,
                      nn_init='glorot_uniform', logger=None, use_sample_weight=False,
                      fl_gamma=0., fl_alpha=1., enable_drm=enable_drm, drm_param=0)
    model.build_model(x=placeholder_x, y=placeholder_y, w=placeholder_w, mode=tf.estimator.ModeKeys.TRAIN)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if enable_drm:
            for _ in range(100):
                _, loss, t0_uplift, t1_uplift, drm = sess.run(
                    [model.train_op, model.loss, model.t0_uplift, model.t1_uplift, model.drm],
                    feed_dict={
                        model.x: data_x,
                        model.y: data_y,
                        model.w: data_w
                    }
                )
                print(loss)
            print(t0_uplift)
            print(data_y[:int(batch_size / 2), :])
            print(t1_uplift)
            print(data_y[int(batch_size / 2):, :])
            print(drm)
        else:
            for _ in range(100):
                _, loss, l0, l1, lg = sess.run([model.train_op, model.loss, model.l0, model.l1, model.lg],
                                               feed_dict={model.x: data_x,
                                                          model.y: data_y,
                                                          model.w: data_w
                                                          })
                print(loss)
            print(l0)
            print(l1)
            print(lg)

