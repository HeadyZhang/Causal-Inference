from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

from utils.tf_utils import *


class T_Learner:
    def __init__(self,
                 x_dim,
                 sparse_x_len,
                 emb_size,
                 layer_units=None,
                 layer_activations=None,

                 fl_gamma=0.0,
                 fl_alpha=1.0,
                 alpha=0.5,
                 beta=0.2,
                 drm_coef=10,
                 base_coef=0.1,

                 lr=0.001,
                 logger=None,
                 bn_decay=0.9,
                 enable_drm=False,
                 use_sample_weight=False,
                 batch_norm=False,
                 ):

        self.logger = logger
        self.x_dim = x_dim
        self.sparse_x_len = sparse_x_len
        self.emb_size = emb_size
        self.lr = lr
        self.enable_drm = enable_drm
        self.batch_norm_decay = bn_decay
        self.batch_norm = batch_norm
        self.use_sample_weight = use_sample_weight
        self.layer_units = layer_units
        self.layer_activations = layer_activations

        self.fl_alpha = fl_alpha
        self.fl_gamma = fl_gamma

        self.alpha = alpha
        self.beta = beta

        self.base_coef = base_coef
        self.drm_coef = drm_coef

        self.x = None
        self.y = None
        self.w = None

        self.y_treatment = None
        self.y_outcome = None

        # self.logits_cr = None
        # self.logits_tr = None

        self.y_cr_pred = None
        self.y_tr_pred = None

        self.loss = None
        self.loss_drm = None
        self.loss_base = None

        self.summary = None
        self.global_step = None
        self.batch_norm_layer = None
        self.train_op = None
        self.ordr1_weights = None
        self.ordr2_emb_mat = None

    def base_model(self, x, is_train):
        if self.batch_norm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()

        with tf.variable_scope('var/fm_first_order', reuse=tf.AUTO_REUSE):
            self.ordr1_weights = tf.get_variable('weight_matrix',
                                                 shape=[self.x_dim, 1],
                                                 dtype=tf.float32,
                                                 initializer=tf.initializers.random_normal(stddev=0.01))
            first_ordr = tf.nn.embedding_lookup(self.ordr1_weights, x)  # [None, sparse_x_len, 1]

            first_ordr = tf.reduce_sum(first_ordr, 2)  # [None, sparse_x_len], deepFM中用于拼接的一阶向量

            intersect = tf.get_variable('intersect', shape=[1], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
            # intersect重复batch_size次后进行拼接
            intersect = tf.tile(intersect[tf.newaxis, :], [tf.shape(first_ordr)[0], 1])  # [None, 1]
            first_ordr = tf.concat([first_ordr, intersect], axis=1)  # [None, sparse_x_len + 1] 截距项也加入一阶向量

        with tf.variable_scope('var/fm_second_order', reuse=tf.AUTO_REUSE):
            self.ordr2_emb_mat = tf.get_variable('embedding_matrix',
                                                 shape=[self.x_dim, self.emb_size],
                                                 dtype=tf.float32,
                                                 initializer=tf.initializers.random_normal(stddev=0.01))
            # 使用最大的index来作为空值
            # ordr2_emb_mat = tf.concat([self.ordr2_emb_mat, tf.zeros(shape=[1, self.emb_size])],
            #                           axis=0)  # [x_dim + 1, emb_size]
            dis_embs = tf.nn.embedding_lookup(self.ordr2_emb_mat, x)  # [None, sparse_x_len, emb_size]
            # sum -> sqr part
            sum_dis_embs = tf.reduce_sum(dis_embs, 1)  # [None, emb_size]
            sum_sqr_dis_embs = tf.square(sum_dis_embs)  # [None, emb_size]
            # sqr -> sum part
            sqr_dis_embs = tf.square(dis_embs)  # [None, sparse_x_len, emb_size]
            sqr_sum_dis_embs = tf.reduce_sum(sqr_dis_embs, 1)  # [None, emb_size]
            # second order
            second_ordr = 0.5 * (sum_sqr_dis_embs - sqr_sum_dis_embs)  # [None, emb_size]

        out_tensors = [first_ordr, second_ordr]  # shape = [None, sparse_x_len + 1 + emb_size]
        output = tf.concat(out_tensors, 1)
        logits = build_mlp_network(input_tensor=output,
                                 layer_units=self.layer_units,
                                 layer_activations=self.layer_activations,
                                 name='nn_network',
                                 bn_layer=self.batch_norm_layer,
                                 disable_last_bn=True,
                                 )
        pred_cr = tf.reshape(logits[:, 0], (-1, 1))
        pred_tr = tf.reshape(logits[:, 1], (-1, 1))

        return pred_cr, pred_tr

    def forward(self, is_train=True):

        return self.base_model(self.x, is_train)

    def cal_loss(self):
        # var_list = tf.trainable_variables(scope='train')
        # _var_vector = tf.concat([tf.reshape(v, shape=(-1,)) for v in var_list], axis=0)
        # if self.mode == tf.estimator.ModeKeys.TRAIN:
        #     for v in var_list:
        #         self.print('reg var: %s' % v)
        #         self.print('*' * 50)
        #     self.print('参数量: %s' % _var_vector.shape)
        self.y_outcome, self.y_treatment = tf.reshape(self.y[:, 0], (-1, 1)), tf.reshape(self.y[:, 1], (-1, 1))
        if self.use_sample_weight:
            w_outcome, w_treatment = tf.reshape(self.w[:, 0], (-1, 1)), tf.reshape(self.w[:, 1], (-1, 1))
        else:
            w_outcome, w_treatment = 1, 1

        # drm loss
        # if self.enable_drm:
        self.loss_base = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(self.y_outcome, self.y_cr_pred) * (1 - self.y_treatment) * w_outcome
            + tf.keras.losses.binary_crossentropy(self.y_outcome, self.y_tr_pred) * self.y_treatment * w_outcome
        )

        if self.enable_drm:
            ite_score = self.alpha * (self.y_tr_pred - self.y_cr_pred)
            self.loss_drm = 1 - drm(self.y_outcome, self.y_treatment, ite_score, with_th=False)
            sample_loss = self.loss_base * self.base_coef + self.loss_drm * self.drm_coef
        else:
            sample_loss = self.loss_base * self.base_coef

        self.loss = tf.reduce_mean(sample_loss)

    def print(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def build_model(self, x=None, y=None, w=None, mode=None, hooks=None):
        self.mode = mode
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
            self.x = tf.reshape(x, shape=(-1, self.sparse_x_len))

            if y is not None:
                self.y = tf.cast(tf.reshape(y, (-1, 2)), dtype=tf.float32)

        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.y_cr_pred, self.y_tr_pred = self.forward(is_train=True)
            else:
                self.y_cr_pred, self.y_tr_pred = self.forward(is_train=False)

            if mode == tf.estimator.ModeKeys.PREDICT:
                pass
            else:
                self.w = w
                self.cal_loss()
                self.summary = tf.summary.scalar('loss', self.loss)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # 训练过程
                if mode == tf.estimator.ModeKeys.TRAIN:
                    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
                    spec = tf.estimator.EstimatorSpec(
                        mode=mode,
                        loss=self.loss,
                        train_op=self.train_op,
                        training_hooks=hooks
                    )
                    tf.summary.scalar('loss', self.loss)
                elif mode == tf.estimator.ModeKeys.PREDICT:
                    out_dict = {'prediction': tf.concat([self.y_cr_pred, self.y_tr_pred], axis=1)}

                    export_outputs = {
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            tf.estimator.export.PredictOutput(out_dict)}
                    spec = tf.estimator.EstimatorSpec(mode=mode,
                                                      predictions=out_dict,
                                                      export_outputs=export_outputs)

                elif mode == tf.estimator.ModeKeys.EVAL:
                    eval_metric_ops = {'loss_base': tf.metrics.mean(self.loss_base),
                                       'loss_drm': tf.metrics.mean(self.loss_drm),
                                       'tr_auc': tf.metrics.auc(self.y[:, 0], self.y_cr_pred),
                                       'cr_auc': tf.metrics.auc(self.y[:, 0], self.y_tr_pred)
                                       }

                    # eval_metric_ops = {}
                    # infer = tf.concat([self.y_cr_pred, self.y_tr_pred], axis=1)
                    # df_infer = pd.DataFrame(infer, columns=[0, 1])
                    # df_infer['y_outcome'] = self.y_outcome
                    # df_infer['treatment'] = self.y_treatment
                    # df_infer['delta'] = df_infer[1] - df_infer[0]
                    #
                    # df1 = df_infer[df_infer['treatment'] == 1]
                    # df0 = df_infer[df_infer['treatment'] == 0]
                    #
                    # l1, l9, dc, auuc, num0, num1 = get_auuc2(df1, df0, 'delta', 'y_outcome', itervals=30)
                    # eval_metric_ops['auuc'] = auuc
                    spec = tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, eval_metric_ops=eval_metric_ops)

                else:
                    raise ValueError('invalid mode: %s' % mode)

                return spec


if __name__ == '__main__':
    layer_units = [32, 32, 2]
    layer_activitions = ['relu', 'relu', 'sigmoid']
    batch_size, x_dim, sparse_x_len, emb_size = 3, 100, 20, 5
    data_x = np.random.random_integers(0, x_dim - 1, size=(batch_size, sparse_x_len))

    data_y = [[1, 0],
              [1, 1],
              [0, 1]]

    data_w = np.random.random_integers(1, 5, size=(batch_size, 2))

    placeholder_x = tf.placeholder(dtype=tf.int64, shape=(None, sparse_x_len))
    placeholder_y = tf.placeholder(dtype=tf.int64, shape=(None, 2))
    placeholder_w = tf.placeholder(dtype=tf.float32, shape=(None, 2))

    model = T_Learner(x_dim,
                      sparse_x_len,
                      emb_size,
                      layer_units=layer_units,
                      layer_activations=layer_activitions,
                      fl_gamma=0.0,
                      fl_alpha=1.0,
                      alpha=0.5,
                      beta=0.2,
                      drm_coef=10,
                      base_coef=0.1,

                      lr=0.001,
                      logger=None,
                      bn_decay=0.9,
                      enable_drm=False,
                      use_sample_weight=True,
                      batch_norm=False)

    model.build_model(x=placeholder_x, y=placeholder_y, w=placeholder_w, mode=tf.estimator.ModeKeys.TRAIN)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for _ in range(100):
            _, loss, l0, l1 = sess.run([model.train_op, model.loss, model.y_tr_pred, model.y_cr_pred],
                                       feed_dict={model.x: data_x,
                                                  model.y: data_y,
                                                  model.w: data_w
                                                  })
            print(loss)
        print(l0)
        print(l1)
















