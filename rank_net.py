from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from utils.tf_utils import *


class RankNet:

    def __init__(self,
                 x_dim,
                 sparse_x_len,
                 emb_size=0,
                 lr=0.001,
                 batch_norm=False,
                 bn_decay=0.9,
                 alpha=0,
                 beta=0,
                 multi_hot_dict=None,
                 logger=None,
                 use_sample_weight=False,
                 fl_gamma=0.,
                 fl_alpha=1.,
                 layer_units=None,
                 layer_activations=None,
                 modifier=1.
                 ):
        # logger
        self.logger = logger

        # 模型参数设置
        self.x_dim = x_dim
        self.sparse_x_len = sparse_x_len
        self.emb_size = emb_size
        self.lr = lr
        self.batch_norm = batch_norm
        self.batch_norm_decay = bn_decay
        self.alpha = alpha
        self.beta = beta
        self.multi_hot_dict = multi_hot_dict if multi_hot_dict is not None else {}
        self.use_sample_weight = use_sample_weight
        self.mode = None
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        # multi-hot在输入x中对应的下标
        i_multi_hot = []
        for key in self.multi_hot_dict:
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
        self.modifier = modifier

        # 模型重要tensor声明
        self.x0 = None
        self.x1 = None
        self.y = None
        self.p_value = None
        self.loss = None
        self.global_step = None
        self.train_op = None
        self.ordr1_weights = None
        self.ordr2_emb_mat = None
        self.sample_weight = None
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

    def base_network(self, x, is_train):
        # DeepFM
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
            self.fm_first_ordr = tf.reduce_sum(first_ordr, axis=1, keepdims=True)
            pred = self.fm_first_ordr + intersect

        if self.emb_size > 0:
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
                second_ordr = 0.5 * (sum_sqr_dis_embs - sqr_sum_dis_embs)  # [None, emb_size], deepFM中用于拼接的二阶向量
                self.fm_second_ordr = tf.reduce_sum(second_ordr, axis=1, keepdims=True)
                pred += self.fm_second_ordr

        if self.layer_units is not None and self.layer_activations is not None:
            # 神经网络参数非空, 进行deepfm
            with tf.variable_scope('embedding_layer', reuse=tf.AUTO_REUSE):
                # 使用fm的embedding作为nn的输入层
                emb_list = []
                one_hot_embs = []  # one-hot特征对应的embedding list, 每一个元素的shape=[None, emb_size]
                for fi in self.i_one_hot:
                    one_hot_embs.append(dis_embs[:, fi])  # [None, emb_size]

                multi_hot_embs = []  # multi-hot特征对应的embedding list, 每一个元素的shape=[None, emb_size]
                for fi in self.multi_hot_dict.keys():
                    f_size = self.multi_hot_dict[fi]
                    f_emb = dis_embs[:, fi: fi + f_size]  # [None, f_size, emb_size]
                    # 平均所有域indices的embeddings得到multi-hot域的embedding
                    f_emb = tf.reduce_mean(f_emb, axis=1)  # [None, emb_size]
                    multi_hot_embs.append(f_emb)

                emb_list.extend(one_hot_embs)
                emb_list.extend(multi_hot_embs)

                self.emb_layer = tf.concat(emb_list, axis=1, name='fm_emb_layer')  # [None, emb_size * field_num]

            with tf.variable_scope('var', reuse=tf.AUTO_REUSE):
                dense_output = get_mlp_network(
                    self.emb_layer,
                    self.layer_units,
                    self.layer_activations,
                    'dense_nn',
                    bn_layer=self.batch_norm_layer,
                    disable_last_bn=False
                )

                # prediction ingredient
                pred += dense_output
        return tf.sigmoid(pred)

    def forward(self, is_train):
        self.pred_0 = self.base_network(self.x0, is_train)        # (None, 1)
        self.pred_1 = self.base_network(self.x1, is_train)        # (None, 1)
        return tf.sigmoid(self.modifier * (self.pred_0 - self.pred_1))      # (None, 1)

    def cal_loss(self):
        # use focal loss
        sample_loss = focal_loss(
            tf.cast(tf.reshape(self.y, [-1]), dtype=tf.float32),
            tf.reshape(self.p_value, [-1]),
            gamma=self.fl_gamma,
            alpha=self.fl_alpha
        )
        if self.sample_weight is not None:
            sample_loss *= self.sample_weight
        sample_loss = tf.reduce_mean(sample_loss)

        # 所有参数组成的向量，用以添加正则项
        var_list = tf.trainable_variables(scope='train/var')
        _var_vector = tf.concat([tf.reshape(v, shape=(-1,)) for v in var_list], axis=0)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            for v in var_list:
                self.print('reg var: %s' % v)
                self.print('*' * 50)
            self.print('参数量: %s' % _var_vector.shape)
        self.param_num = _var_vector.shape
        reg_loss = self.alpha * (self.beta * tf.reduce_sum(tf.abs(_var_vector))
                                 + (1 - self.beta) * tf.reduce_sum(tf.square(_var_vector)))
        loss = sample_loss + reg_loss
        return loss, reg_loss

    def build_model(self, x0=None, x1=None, y=None, w=None, mode=None, hooks=None):
        """

        :param x0: tensor类型, 稀疏表达的multi-hot向量, 包含所有向量值为1的indices(展示情况)
        :param x1: tensor类型, 稀疏表达的multi-hot向量, 包含所有向量值为1的indices(不展示情况)
        :param y: tensor类型, label, 展示情况更好则为1, 否则为0
        :param w: tensor类型, sample weight
        :param mode: tf.estimator.ModeKeys的某个枚举值, 用于指定模型处于训练、评估或是推断模式
        :param hooks: TensorFlow hook
        :return:
        """
        self.mode = mode
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
            self.x0 = tf.reshape(x0, shape=(-1, self.sparse_x_len))  # 任何情况下x都不能为空
            self.x1 = tf.reshape(x1, shape=(-1, self.sparse_x_len))

            if y is not None:
                # 对于推理过程, y可以为空
                self.y = tf.cast(tf.reshape(y, (-1, 1)), dtype=tf.float32)

        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.p_value = self.forward(is_train=True)
            else:
                self.p_value = self.forward(is_train=False)

            if mode == tf.estimator.ModeKeys.PREDICT:
                # 对于推理过程, 不应计算loss, 否则会强制要求输入y
                pass
            else:
                # 非推理过程, 需要计算loss
                self.sample_weight = w if self.use_sample_weight else None
                self.loss, self.reg_loss = self.cal_loss()
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
                    out_dict = {'prediction': self.p_value}  # shape=[None, 1]
                    export_outputs = {
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            tf.estimator.export.PredictOutput(out_dict)}
                    spec = tf.estimator.EstimatorSpec(mode=mode,
                                                      predictions=out_dict,
                                                      export_outputs=export_outputs)
                elif mode == tf.estimator.ModeKeys.EVAL:
                    # 评估过程
                    eval_metric_ops = {
                        'reg_loss': tf.metrics.mean(self.reg_loss),
                        'auc': tf.metrics.auc(self.y, self.p_value)
                    }

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
    # mock data
    batch_size, x_dim, sparse_x_len, emb_size = 20, 100, 10, 5
    layer_units = [128, 64, 1]
    layer_activations = ['relu', 'relu', 'relu']

    data_x0 = np.random.random_integers(0, x_dim - 1, size=(batch_size, sparse_x_len))
    data_x1 = np.random.random_integers(0, x_dim - 1, size=(batch_size, sparse_x_len))
    data_y = np.random.random_integers(0, 1, size=(batch_size, 1))
    data_w = np.ones(shape=(batch_size, 1))

    # place holders
    placeholder_x0 = tf.placeholder(dtype=tf.int64, shape=(None, sparse_x_len))
    placeholder_x1 = tf.placeholder(dtype=tf.int64, shape=(None, sparse_x_len))
    placeholder_y = tf.placeholder(dtype=tf.int64, shape=(None, 1))
    placeholder_w = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    model = RankNet(x_dim, sparse_x_len, emb_size=emb_size, batch_norm=True, use_sample_weight=True,
                    layer_units=layer_units, layer_activations=layer_activations, modifier=1)
    model.build_model(
        x0=placeholder_x0,
        x1=placeholder_x1,
        y=placeholder_y,
        w=placeholder_w,
        mode=tf.estimator.ModeKeys.TRAIN
    )
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(100):
            _, loss = sess.run([model.train_op, model.loss],
                               feed_dict={
                                   model.x0: data_x0, model.x1: data_x1,
                                   model.y: data_y, model.sample_weight: data_w
                               })
            print(loss)
        print(sess.run(model.pred_0, feed_dict={model.x0: data_x0}))
        print(sess.run(model.pred_1, feed_dict={model.x1: data_x1}))
