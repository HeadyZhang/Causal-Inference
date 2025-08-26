#-*- coding : utf8 -*-

from utils.tf_utils import *
from utils.modules import *


class TransformerModel:
    def __init__(self, x_dim, sparse_x_len,
                 id_emb_size, x_emb_size,
                  u_seq_len,
                 hidden_size=768,
                 d_ff=2048,
                 hidden_units=[32, 16, 1],
                 layer_activation=['relu','relu','sigmoid'],
                 i_hash_size=500000,
                 c_hash_size=1000,
                 fl_gamma=0.0,
                 fl_alpha=1.0,
                 lr=0.001,
                 keep_pro=0.9,
                 num_blocks=2,
                 num_heads=8,
                 initializer_range=0.02,
                 batch_norm=False,
                 use_sample_weight=False,
                 logger=None
                 ):

        self.x_dim = x_dim
        self.sparse_x_len = sparse_x_len
        self.id_emb_size = id_emb_size
        self.x_emb_size = x_emb_size
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.use_sample_weight = use_sample_weight
        self.hidden_units = hidden_units
        self.layer_activation = layer_activation
        self.u_seq_len = u_seq_len
        self.i_hash_size = i_hash_size
        self.c_hash_size = c_hash_size
        self.lr = lr
        self.fl_alpha = fl_alpha
        self.fl_gamma = fl_gamma
        self.logger = logger
        self.drop_out = 1 - keep_pro
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.initializer_range = initializer_range



        self.mode = None

        # 模型重要的tensor声明
        self.x = None
        self.item_id = None
        self.cate_id = None
        self.label = None
        self.pred = None

        self.u_item_seq = None
        self.u_cate_seq = None

        self.sample_weight = None
        self.summary = None
        self.loss = None

        self.global_step = None
        self.train_op = None

        self.item_batch_emb = None
        self.cate_batch_emb = None
        self.u_clk_item_seq_batch_emb = None
        self.item_seq_len = None

    def train_bn_layer(self):
        pass

    def inf_bn_layer(self):
        pass

    def forword(self, is_train):
        with tf.variable_scope("var/embedding_layer", reuse=tf.AUTO_REUSE):
            # item id --> item hash id
            item_hash_column = tf.feature_column.categorical_column_with_hash_bucket(key='item_id',
                                                                                     hash_bucket_size=self.i_hash_size,
                                                                                     dtype=tf.int32)
            item_embedding_column = tf.feature_column.embedding_column(categorical_column=item_hash_column,
                                                                       dimension=self.id_emb_size,
                                                                       initializer=tf.initializers.truncated_normal(stddev=self.initializer_range),
                                                                       trainable=True)
            self.item_batch_emb = tf.feature_column.input_layer(
                {'item_id': self.item_id}, [item_embedding_column]
            )                                           # shape=[None, None, emb_size]

            # cate id --> cate hash id
            cate_hash_column = tf.feature_column.categorical_column_with_hash_bucket(key='cate_id',
                                                                                     hash_bucket_size=self.c_hash_size,
                                                                                     dtype=tf.int32)

            cate_embedding_column = tf.feature_column.embedding_column(categorical_column=cate_hash_column,
                                                                       dimension=self.id_emb_size,
                                                                       initializer=tf.initializers.truncated_normal(stddev=self.initializer_range),
                                                                       trainable=True)
            self.cate_batch_emb = tf.feature_column.input_layer(
                {'cate_id': self.cate_id}, [cate_embedding_column]
            )

            # seq id --> seq hash id --> emb
            u_clk_item_seq_hash_column = tf.contrib.feature_column.sequence_categorical_column_with_hash_bucket(key='u_clk_item_seq',
                                                                                                                hash_bucket_size=self.i_hash_size,
                                                                                                                dtype=tf.int32
            )
            u_clk_item_seq_embedding = tf.feature_column.embedding_column(categorical_column=u_clk_item_seq_hash_column,
                                                                          dimension=self.id_emb_size,
                                                                          initializer=tf.initializers.truncated_normal(stddev=self.initializer_range),
                                                                          trainable=True)

            self.u_clk_item_seq_batch_emb, self.item_seq_len = tf.contrib.feature_column.sequence_input_layer(
                {'u_clk_item_seq': self.u_clk_item_seq}, [u_clk_item_seq_embedding])

            u_clk_cate_seq_hash_column = tf.contrib.feature_column.sequence_categorical_column_with_hash_bucket(key='u_clk_cate_seq',
                                                                                                                hash_bucket_size=self.c_hash_size,
                                                                                                                dtype=tf.int32)
            u_clk_cate_seq_embedding = tf.feature_column.embedding_column(categorical_column=u_clk_cate_seq_hash_column,
                                                                          dimension=self.id_emb_size,
                                                                          initializer=tf.initializers.truncated_normal(stddev=self.initializer_range),
                                                                          trainable=True)
            self.u_clk_cate_seq_batch_emb, _ = tf.contrib.feature_column.sequence_input_layer(
                {'u_clk_cate_seq': self.u_clk_cate_seq}, [u_clk_cate_seq_embedding])

            # u_noclk_item_seq_hash_column = tf.contrib.feature_column.sequence_categorical_column_with_hash_bucket(
            #     'u_noclk_item_seq',
            #     self.i_hash_size,
            #     dtype=tf.int32
            # )
            # u_noclk_item_seq_embedding = tf.feature_column.embedding_column(u_noclk_item_seq_hash_column,
            #                                                                 self.id_emb_size,
            #                                                                 trainable=True)
            # self.u_noclk_item_seq_batch_emb, _ = tf.contrib.feature_column.sequence_input_layer(
            #     {'u_noclk_item_seq':  self.u_noclk_item_seq}, [u_noclk_item_seq_embedding]
            # )
            #
            # u_noclk_cate_seq_hash_column = tf.contrib.feature_column.sequence_categorical_column_with_hash_bucket(
            #     'u_noclk_cate_seq', self.c_hash_size, dtype=tf.int32
            # )
            # u_noclk_cate_seq_embedding = tf.feature_column.embedding_column(u_noclk_cate_seq_hash_column, self.id_emb_size, trainable=True)
            # self.u_noclk_cate_seq_batch_emb, _ = tf.contrib.feature_column.sequence_input_layer(
            #     {'u_noclk_cate_seq': self.u_noclk_cate_seq}, [u_noclk_cate_seq_embedding]
            # )

            self.ordr2_emb_mat = tf.get_variable('embedding_matrix',
                                                 shape=[self.x_dim, self.x_emb_size],
                                                 dtype=tf.float32,
                                                 initializer=tf.initializers.truncated_normal(stddev=0.02))
            dis_embs = tf.nn.embedding_lookup(self.ordr2_emb_mat, self.x)  # [None, sparse_x_len, emb_size]
            # 通过embedding将输入维度从x_dim降至sparse_x_len * emb_size
            dense_x = tf.reshape(dis_embs, shape=(-1, self.sparse_x_len * self.x_emb_size))


        self.item_emb = tf.concat([self.item_batch_emb, self.cate_batch_emb], 1)  # [None, emb_size * 2]
        self.u_clk_item_his_emb = tf.concat([self.u_clk_item_seq_batch_emb, self.u_clk_cate_seq_batch_emb], 2)   # [None, None, emb_size * 2]

        # with tf.variable_scope('var/fm_first_order', reuse=tf.AUTO_REUSE):
        #     self.ordr1_weights = tf.get_variable('weight_matrix',
        #                                          shape=[self.x_dim, 1],
        #                                          dtype=tf.float32,
        #                                          initializer=tf.initializers.truncated_normal(stddev=0.02))
        #     first_ordr = tf.nn.embedding_lookup(self.ordr1_weights, self.x)  # [None, sparse_x_len, 1]
        #     first_ordr = tf.reduce_sum(first_ordr, 2)  # [None, sparse_x_len], deepFM中用于拼接的一阶向量
        #     intersect = tf.get_variable('intersect', shape=[1], dtype=tf.float32,
        #                                 initializer=tf.zeros_initializer())
        #
        #     intersect = tf.tile(intersect[tf.newaxis, :], [tf.shape(first_ordr)[0], 1])  # [None, 1]
        #
        #     first_ordr = tf.concat([first_ordr, intersect], axis=1)  # [None, sparse_x_len + 1] 截距项也加入一阶向量
        #
        # with tf.variable_scope('var/fm_second_order', reuse=tf.AUTO_REUSE):
        #
        #     self.ordr2_emb_mat = tf.get_variable('embedding_matrix',
        #                                          shape=[self.x_dim, self.x_emb_size],
        #                                          dtype=tf.float32,
        #                                          initializer=tf.initializers.truncated_normal(stddev=0.02))
        #     dis_embs = tf.nn.embedding_lookup(self.ordr2_emb_mat, self.x)  # [None, sparse_x_len, emb_size]
        #
        #     # sum -> sqr part
        #     sum_dis_embs = tf.reduce_sum(dis_embs, 1)  # [None, emb_size]
        #     sum_sqr_dis_embs = tf.square(sum_dis_embs)  # [None, emb_size]
        #     # sqr -> sum part
        #     sqr_dis_embs = tf.square(dis_embs)  # [None, sparse_x_len, emb_size]
        #     sqr_sum_dis_embs = tf.reduce_sum(sqr_dis_embs, 1)  # [None, emb_size]
        #     # second order
        #     second_ordr = 0.5 * (sum_sqr_dis_embs - sqr_sum_dis_embs)  # [None, emb_size]
        #
        # dense_x = tf.concat([first_ordr, second_ordr], 1)  # shape = [None, sparse_x_len + 1 + emb_size]

        with tf.variable_scope("var/bst_encoder", reuse=tf.AUTO_REUSE):
            ## 目标id和序列id一起输入
            target_item_enc = tf.expand_dims(self.item_emb, axis=1)
            enc = self.u_clk_item_his_emb
            enc = tf.concat([enc, target_item_enc], 1)

            enc *= self.id_emb_size ** 0.5
            enc += positional_encoding(enc, self.u_seq_len + 1)

            # layer norm  & dropout
            # enc = tf.contrib.layers.layer_norm(
            #     inputs=enc, begin_norm_axis=-1, begin_params_axis=-1, scope="emb_layer_norm")

            enc = tf.layers.dense(enc, self.hidden_size,
                                  kernel_initializer=tf.initializers.truncated_normal(stddev=0.02))

            enc = tf.layers.dropout(enc, rate=self.drop_out, training=(self.mode == tf.estimator.ModeKeys.TRAIN))



            src_mask = tf.equal(tf.concat([self.u_clk_item_seq, self.item_id], 1), 0)

            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_mask,
                                              num_heads=self.num_heads,
                                              keep_rate=self.drop_out,
                                              causality=False,
                                              initializer_range=self.initializer_range,
                                              is_train=is_train)
                    # Feed forward
                    enc = ff(enc, num_units=[self.d_ff, self.hidden_size])

        # with tf.variable_scope("var/bst_decoder", reuse=tf.AUTO_REUSE):
        #     pass

        with tf.variable_scope("maxPool", reuse=tf.AUTO_REUSE):
            outputs = tf.reduce_max(enc, 1)

        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            outputs = tf.reshape(outputs, [-1, self.hidden_size])
            inputs = tf.concat([outputs, dense_x], 1)
            net = tf.layers.dropout(inputs, rate=self.drop_out, training=(self.mode==tf.estimator.ModeKeys.TRAIN))
            for layer_id, num_hidden_units in enumerate(self.hidden_units[0:-1]):
                net = tf.layers.dense(net, units=num_hidden_units,
                                      activation=self.layer_activation[layer_id],
                                      kernel_initializer=tf.initializers.truncated_normal(stddev=0.02),
                                      name="dense_layer_%d" % (layer_id)
                                      )
                if self.batch_norm:
                    net = tf.layers.batch_normalization(net, center=True, scale=True,
                                                        training=(self.mode == tf.estimator.ModeKeys.TRAIN),
                                                        trainable=is_train)

                net = tf.layers.dropout(net, rate=self.drop_out, training=(self.mode == tf.estimator.ModeKeys.TRAIN))

            dense_output = tf.layers.Dense(units=self.hidden_units[-1],
                                           activation=self.layer_activation[-1],
                                           kernel_initializer=tf.initializers.truncated_normal(stddev=0.02),
                                           name='dense_layer_%d' % (len(self.hidden_units) - 1),
                                           )(net)


            return dense_output

    def cal_loss(self, pred, y, w):
        # shape=[None,]
            # use focal loss
        sample_loss = focal_loss(
                tf.cast(tf.reshape(y, [-1]), dtype=tf.float32),
                tf.reshape(pred, [-1]),
                gamma=self.fl_gamma,
                alpha=self.fl_alpha
            )

        if w is not None:
            sample_loss *= w  # shape=[None,]
        sample_loss = tf.reduce_mean(sample_loss)  # shape=[] (scalar)

        return sample_loss

    def build_model(self, x, item_id, cate_id, u_clk_item_seq, u_clk_cate_seq,
                    label=None, w=None, mode=None, hooks=None):
        self.mode = mode
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
            self.x = tf.reshape(x, shape=(-1, self.sparse_x_len))
            self.item_id = item_id
            self.cate_id = cate_id
            self.u_clk_item_seq = u_clk_item_seq
            self.u_clk_cate_seq = u_clk_cate_seq

            if label is not None:
                self.label = tf.cast(label, dtype=tf.float32)

        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.pred = self.forword(is_train=True)
            else:
                self.pred = self.forword(is_train=False)

            if mode == tf.estimator.ModeKeys.PREDICT:
                pass
            else:
                # 非推理过程
                self.sample_weight = w if self.use_sample_weight else None
                self.loss = self.cal_loss(self.pred, self.label, self.sample_weight)
                self.summary = tf.summary.scalar('loss', self.loss)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if mode == tf.estimator.ModeKeys.TRAIN:
                    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, self.global_step)

                    spec = tf.estimator.EstimatorSpec(
                        mode=mode,
                        loss=self.loss,
                        train_op=self.train_op,
                        training_hooks=hooks
                    )
                    tf.summary.scalar('loss', self.loss)
                elif mode == tf.estimator.ModeKeys.PREDICT:
                    out_dict = {'prediction': self.pred}
                    export_outputs = {
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            tf.estimator.export.PredictOutput(out_dict)
                    }

                    spec = tf.estimator.EstimatorSpec(mode=mode,
                                                      predictions=out_dict,
                                                      export_outputs=export_outputs)
                elif mode == tf.estimator.ModeKeys.EVAL:
                    eval_metric_ops = {
                        # 'aux_loss': tf.metrics.mean(self.aux_loss)
                    }
                    eval_metric_ops['pred_auc'] = tf.metrics.auc(self.label, self.pred)
                    eval_metric_ops['pred_acc'] = tf.metrics.accuracy(labels=self.label,
                                                                      predictions=tf.to_float(tf.greater_equal(self.pred, 0.5)))

                    spec = tf.estimator.EstimatorSpec(
                        mode=mode,
                        loss=self.loss,
                        eval_metric_ops=eval_metric_ops
                    )

                else:
                    raise ValueError('invalid mode: %s'% mode)
                return spec

    def print(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)


if __name__ == '__main__':
    batch_size, x_dim, sparse_x_len, id_emb_size, x_emb_size = 3, 100, 10, 16, 5

    data_x = np.random.random_integers(0, x_dim - 1, size=(batch_size, sparse_x_len))

    pgc_id = [[23445556], [3345557], [888899]]
    cate_id = [[1],[2], [3]]
    u_clk_item_seq = [
            [3446677, 3344009, 4455624],
            [77899888, 99009988, 990999],
            [778877, 7776668877, 55343355]
    ]
    u_clk_cate_seq = [
                      [1,2,4],
                      [8,9,10],
                      [11,12,13]]

    u_noclk_item_seq = [
        [33445, 55566, 0],
        [33475, 66666, 88889],
        [99009, 89056, 899889]
    ]

    u_noclk_cate_seq = [
        [3,5,6],
        [7,10,11],
        [20,21,22]
    ]
    data_w = np.random.standard_normal(size=(batch_size, 1))
    data_label = np.random.random_integers(0, 1, size=(batch_size, 1))
    print(data_label)

    placeholder_x = tf.placeholder(dtype=tf.int64, shape=(None, sparse_x_len))

    placeholder_item_id = tf.placeholder(dtype=tf.int64, shape=(None, 1))
    placeholder_cate_id = tf.placeholder(dtype=tf.int64, shape=(None, 1))

    placeholder_item_seq = tf.placeholder(dtype=tf.int64, shape=(None, 3))
    placeholder_cate_seq = tf.placeholder(dtype=tf.int64, shape=(None, 3))
    placeholder_mask_seq = tf.placeholder(dtype=tf.float32, shape=(None, 3))
    placeholder_label = tf.placeholder(dtype=tf.int64, shape=(None, 1))
    placeholder_w = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    placeholder_no_item_seq = tf.placeholder(dtype=tf.int64, shape=(None, 3))
    placeholder_no_cate_seq = tf.placeholder(dtype=tf.int64, shape=(None, 3))

    model = TransformerModel(
                            x_dim=x_dim,
                            sparse_x_len=sparse_x_len,
                            id_emb_size=id_emb_size,
                            x_emb_size=x_emb_size,
                            hidden_size=16,
                            hidden_units=[32, 16, 1],
                            u_seq_len=3,
                            i_hash_size=50,
                            c_hash_size=10,
                            fl_gamma=0,
                            fl_alpha=1.0,
                            lr=0.001,
                            batch_norm=True,
                            use_sample_weight=True
                        )

    model.build_model(
                      x=placeholder_x,
                      item_id=placeholder_item_id,
                      cate_id=placeholder_cate_id,
                      u_clk_item_seq=placeholder_item_seq,
                      u_clk_cate_seq=placeholder_cate_seq,
                      label=placeholder_label,
                      w=placeholder_w,
                      mode=tf.estimator.ModeKeys.TRAIN,
                      hooks=None)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for _ in range(100):
            _, loss = sess.run([model.train_op, model.loss],
                               feed_dict={
                                   model.x: data_x,
                                   model.item_id: pgc_id,
                                   model.cate_id: cate_id,
                                   model.u_clk_item_seq: u_clk_item_seq,
                                   model.u_clk_cate_seq: u_clk_cate_seq,
                                   model.label: data_label,
                                   model.sample_weight: data_w
                               })

            print(loss)
        print(sess.run(model.pred, feed_dict={
            model.x: data_x,
            model.item_id: pgc_id,
            model.cate_id: cate_id,
            model.u_clk_item_seq: u_clk_item_seq,
            model.u_clk_cate_seq: u_clk_cate_seq

        }))


