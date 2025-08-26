import tensorflow as tf
import numpy as np



class EUEN:
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
                 iter_step=100,
                 base_model='fm',
                 enable_values=False):
        self.logger = logger
        self.x_dim = x_dim
        self.sparse_x_len = sparse_x_len
        self.emb_size = emb_size
        self.lr = lr
        self.mode = None
        self.batch_norm = batch_norm
        self.base_model = base_model
        self.use_sample_weight = use_sample_weight
        self.every_n_iter = iter_step
        self.enable_values = enable_values



        self.layer_units = [int(x) for x in layer_units.split(',')] if type(layer_units) is not list \
            else layer_units

        self.layer_activations = list(
            map(lambda x: None if (x == 'None') else x, layer_activations.split(','))) if type(
            layer_activations) is not list \
            else layer_activations

        self.y_treatment = None
        self.y_outcome = None

        self.loss = None

        self.y_cr_pred = None
        self.y_tr_pred = None

        self.global_step = None

        self.x_indices = None
        self.x_values = None

        self.y = None
        self.w = None

        self.c_logit = None
        self.c_tau = None
        self.c_prob = None
        self.t_logit = None
        self.u_tau = None
        self.t_prob = None
        self.prob = None

    def build_mlp_network(self, input_tensor, layer_units, layer_activations, is_train, disable_last_bn=False):
        input_x = input_tensor
        for i, (units, activation) in enumerate(zip(layer_units, layer_activations)):
            input_x = tf.keras.layers.Dense(units, activation=activation)(input_x)
            if self.batch_norm:
                if disable_last_bn and i == len(layer_units) - 1:
                    continue
                input_x = tf.keras.layers.BatchNormalization(trainable=is_train)(input_x)

        return input_x

    def ControlNet(self, x, is_training):

        with tf.compat.v1.variable_scope('var/fm_first_order', reuse=tf.compat.v1.AUTO_REUSE):
            self.ordr1_weights = tf.compat.v1.get_variable('weight_matrix',
                                                           shape=[self.x_dim, 1],
                                                           dtype=tf.float32,
                                                           initializer=tf.initializers.random_normal(stddev=0.01))
            first_ordr = tf.nn.embedding_lookup(self.ordr1_weights, x)  # [None, sparse_x_len, 1]

            first_ordr = tf.reduce_sum(first_ordr, 2)  # [None, sparse_x_len], deepFM中用于拼接的一阶向量

            intersect = tf.compat.v1.get_variable('intersect', shape=[1], dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
            # intersect重复batch_size次后进行拼接
            intersect = tf.compat.v1.tile(intersect[tf.newaxis, :], [tf.compat.v1.shape(first_ordr)[0], 1])  # [None, 1]
            first_ordr = tf.concat([first_ordr, intersect], axis=1)  # [None, sparse_x_len + 1] 截距项也加入一阶向量

        with tf.compat.v1.variable_scope('var/fm_second_order', reuse=tf.compat.v1.AUTO_REUSE):
            self.ordr2_emb_mat = tf.compat.v1.get_variable('embedding_matrix',
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
        outputs = tf.concat(out_tensors, 1)

        c_logits = self.build_mlp_network(outputs, self.layer_units, self.layer_activations, is_training)
        c_tau = c_logits
        c_prob = tf.nn.sigmoid(c_logits)
        return c_logits, c_tau, c_prob


    def UpliftNet(self, x, is_training):
        with tf.compat.v1.variable_scope('var/fm_first_order', reuse=tf.compat.v1.AUTO_REUSE):
            self.ordr1_weights = tf.compat.v1.get_variable('weight_matrix',
                                                           shape=[self.x_dim, 1],
                                                           dtype=tf.float32,
                                                           initializer=tf.initializers.random_normal(stddev=0.01))
            first_ordr = tf.nn.embedding_lookup(self.ordr1_weights, x)  # [None, sparse_x_len, 1]
            # first_ordr = tf.reshape(first_ordr, shape=(-1, self.sparse_x_len))       # [None, sparse_x_len]

            first_ordr = tf.reduce_sum(first_ordr, 2)  # [None, sparse_x_len], deepFM中用于拼接的一阶向量
            if self.enable_values:
                first_ordr = first_ordr * self.x_values

            intersect = tf.compat.v1.get_variable('intersect', shape=[1], dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
            # self.fm_first_ordr = tf.reduce_sum(first_ordr, axis=1, keepdims=True) + intersect
            # pred = self.fm_first_ordr

            # intersect重复batch_size次后进行拼接
            intersect = tf.compat.v1.tile(intersect[tf.newaxis, :], [tf.compat.v1.shape(first_ordr)[0], 1])  # [None, 1]
            first_ordr = tf.concat([first_ordr, intersect], axis=1)  # [None, sparse_x_len + 1] 截距项也加入一阶向量

        with tf.compat.v1.variable_scope('var/fm_second_order', reuse=tf.compat.v1.AUTO_REUSE):
            self.ordr2_emb_mat = tf.compat.v1.get_variable('embedding_matrix',
                                                           shape=[self.x_dim, self.emb_size],
                                                           dtype=tf.float32,
                                                           initializer=tf.initializers.random_normal(stddev=0.01))
            # 使用最大的index来作为空值
            # ordr2_emb_mat = tf.concat([self.ordr2_emb_mat, tf.zeros(shape=[1, self.emb_size])],
            #                           axis=0)  # [x_dim + 1, emb_size]
            dis_embs = tf.nn.embedding_lookup(self.ordr2_emb_mat, x)  # [None, sparse_x_len, emb_size]
            if self.enable_values:
                sum_dis_embs = tf.reduce_sum(dis_embs * tf.expand_dims(self.x_values, axis=2), 1)  # [None, emb_size]
            else:
                sum_dis_embs = tf.reduce_sum(dis_embs, 1)  # [None, emb_size]
            sum_sqr_dis_embs = tf.square(sum_dis_embs)  # [None, emb_size]

            if self.enable_values:
                sqr_dis_embs = tf.square(
                    dis_embs * tf.expand_dims(self.x_values, axis=2))  # [None, sparse_x_len, emb_size]
            else:
                sqr_dis_embs = tf.square(dis_embs)  # [None, sparse_x_len, emb_size]
            sqr_sum_dis_embs = tf.reduce_sum(sqr_dis_embs, 1)  # [None, emb_size]
            # second order
            second_ordr = 0.5 * (sum_sqr_dis_embs - sqr_sum_dis_embs)  # [None, emb_size]

        out_tensors = [first_ordr, second_ordr]  # shape = [None, sparse_x_len + 1 + emb_size]
        outputs = tf.concat(out_tensors, 1)
        t_logits = self.build_mlp_network(outputs, self.layer_units, self.layer_activations, is_training)
        u_tau = t_logits
        t_prob = tf.nn.sigmoid(t_logits)
        return t_logits, u_tau, t_prob

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

    def forward(self, is_train=True):
        c_logit, c_tau, c_prob = self.ControlNet(self.x_indices, is_train)
        t_logit, u_tau, t_prob = self.UpliftNet(self.x_indices, is_train)
        return c_logit, c_tau, c_prob, t_logit, u_tau, t_prob

    def cal_loss(self):
        self.y_outcome, self.y_treatment = tf.reshape(self.y[:, 0], (-1, 1)), tf.reshape(self.y[:, 1], (-1, 1))
        if self.use_sample_weight:
            w_outcome, w_treatment = tf.reshape(self.w[:, 0], (-1, 1)), tf.reshape(self.w[:, 1], (-1, 1))
        else:
            w_outcome, w_treatment = 1, 1
        loss = tf.keras.losses.binary_crossentropy(self.y_outcome,
                                                     (1 - self.y_treatment) * self.c_prob + self.y_treatment * self.prob) * w_outcome

        return tf.reduce_mean(loss)


    def build_model(self, x_indices=None, x_values=None, y=None, w=None, mode=None):
        self.mode = mode
        with tf.compat.v1.variable_scope('input', reuse=tf.compat.v1.AUTO_REUSE):

            self.x_indices = tf.reshape(x_indices, shape=(-1, self.sparse_x_len))
            if self.enable_values:
                self.x_values = tf.reshape(x_values, shape=(-1, self.sparse_x_len))
                self.x_values = tf.concat([self.x_values[:, 0: 1] / 100, self.x_values[:, 1:]], axis=1)  # 分转元

            if y is not None:
                self.y = tf.cast(tf.reshape(y, (-1, 2)), dtype=tf.float32)


        with tf.compat.v1.variable_scope('train', reuse=tf.compat.v1.AUTO_REUSE):
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.c_logit, self.c_tau, self.c_prob, self.t_logit, self.u_tau, self.t_prob = self.forward(is_train=True)
            else:
                self.c_logit, self.c_tau, self.c_prob, self.t_logit, self.u_tau, self.t_prob = self.forward(is_train=False)
            self.prob = tf.nn.sigmoid(self.c_logit + self.t_logit)

            if mode == tf.estimator.ModeKeys.PREDICT:
                out_dict = {'prediction':  tf.concat([self.c_prob, self.prob], axis=1)}

                export_outputs = {
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(out_dict)
                }

                return tf.estimator.EstimatorSpec(mode=mode,
                                                  predictions=out_dict,
                                                  export_outputs=export_outputs)
            self.w = w if self.use_sample_weight else None

            self.loss = self.cal_loss()
            auc, auc_op = tf.compat.v1.metrics.auc(self.y[:, 0], self.c_prob)
            t_auc, t_auc_op = tf.compat.v1.metrics.auc(self.y[:, 1], self.t_prob)

            tf.compat.v1.summary.scalar('loss', self.loss)

            eval_metric_ops = {
                "auc": (auc, auc_op),
                "t_auc": (t_auc, t_auc_op)
            }

            tensors_to_log = {
                "loss": self.loss,
                "auc": auc_op,
                "t_auc": t_auc_op
            }
            logging_hook = tf.estimator.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=self.every_n_iter)
            hooks = [logging_hook, tf.estimator.StepCounterHook()]
            # 训练过程
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

    layer_units = [32, 32, 1]
    layer_activations = ['relu', 'relu', None]
    batch_size, x_dim, sparse_x_len, emb_size = 3, 100, 20, 5
    data_x = np.random.random_integers(0, x_dim - 1, size=(batch_size, sparse_x_len))

    data_y = [[1, 0],
              [1, 1],
              [0, 1]]

    data_w = np.random.random_integers(1, 5, size=(batch_size, 2))

    placeholder_x = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None, sparse_x_len))
    placeholder_y = tf.compat.v1.placeholder(dtype=tf.int64, shape=(None, 2))
    placeholder_w = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 2))

    model = EUEN(x_dim,
                    sparse_x_len,
                      emb_size,
                      layer_units=layer_units,
                      layer_activations=layer_activations,
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
                      batch_norm=True)

    # model.build_model(x=placeholder_x, y=placeholder_y, w=placeholder_w, mode=tf.estimator.ModeKeys.TRAIN)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        for _ in range(100):
            _, loss, c_prob, t_prob = sess.run([model.train_op, model.loss, model.c_prob, model.t_prob],
                                       feed_dict={model.x: data_x,
                                                  model.y: data_y,
                                                  model.w: data_w
                                                  })
            print(loss)
        print(c_prob)
        print(t_prob)










