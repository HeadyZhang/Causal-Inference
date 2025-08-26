#-*- coding : utf8 -*-

from utils.rnn import dynamic_rnn, VecAttGRUCell
from utils.tf_utils import *
from tensorflow.python.ops.rnn_cell import GRUCell


class DeepInterestEvolutionNet:
    def __init__(self,
                 emb_size,
                 hidden_size,
                 attention_size,
                 u_seq_len,
                 u_hash_size=20000000,
                 i_hash_size=5000000,
                 c_hash_size=1000,
                 fl_gamma=0.0,
                 fl_alpha=1.0,
                 lr=0.001,
                 batch_norm=False,
                 use_sample_weight=False,
                 use_negsampling=False,
                 logger=None
                 ):
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.batch_norm = batch_norm
        self.use_sample_weight = use_sample_weight
        self.use_negsampling = use_negsampling
        self.u_seq_len = u_seq_len
        self.u_hash_size = u_hash_size
        self.i_hash_size = i_hash_size
        self.c_hash_size = c_hash_size
        self.lr = lr
        self.fl_alpha = fl_alpha
        self.fl_gamma = fl_gamma
        self.logger = logger

        self.mode = None

        # 模型重要的tensor声明
        self.uid = None
        self.item_id = None
        self.cate_id = None
        self.label = None
        self.pred = None

        self.u_item_seq = None
        self.u_cate_seq = None

        self.sample_weight = None
        self.summary = None
        self.mask = None
        self.aux_loss = None
        self.loss = None

        self.global_step = None
        self.train_op = None


        self.uid_batch_emb = None
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
            # uid --> new hash id
            uid_hash_column = tf.feature_column.categorical_column_with_hash_bucket('uid',
                                                                                self.u_hash_size,
                                                                                dtype=tf.int32)

            uid_embedding_column = tf.feature_column.embedding_column(uid_hash_column,
                                                                  self.emb_size,
                                                                  trainable=True)

            self.uid_batch_emb = tf.feature_column.input_layer({'uid': self.uid}, [uid_embedding_column])

            # item id --> item hash id
            item_hash_column = tf.feature_column.categorical_column_with_hash_bucket('item_id',
                                                                                     self.i_hash_size,
                                                                                     dtype=tf.int32)
            item_embedding_column = tf.feature_column.embedding_column(item_hash_column,
                                                                       self.emb_size,
                                                                       trainable=True)
            self.item_batch_emb = tf.feature_column.input_layer(
                {'item_id': self.item_id}, [item_embedding_column]
            )                                           # shape=[None, None, emb_size]

            # cate id --> cate hash id
            cate_hash_column = tf.feature_column.categorical_column_with_hash_bucket('cate_id',
                                                                                     self.c_hash_size,
                                                                                     dtype=tf.int32)

            cate_embedding_column = tf.feature_column.embedding_column(cate_hash_column,
                                                                       self.emb_size,
                                                                       trainable=True)
            self.cate_batch_emb = tf.feature_column.input_layer(
                {'cate_id': self.cate_id}, [cate_embedding_column]
            )

            # seq id --> seq hash id --> emb
            u_clk_item_seq_hash_column = tf.contrib.feature_column.sequence_categorical_column_with_hash_bucket(
                'u_clk_item_seq',
                self.i_hash_size,
                dtype=tf.int32
            )
            u_clk_item_seq_embedding = tf.feature_column.embedding_column(u_clk_item_seq_hash_column,
                                                                          self.emb_size,
                                                                          trainable=True)

            self.u_clk_item_seq_batch_emb, self.item_seq_len = tf.contrib.feature_column.sequence_input_layer(
                {'u_clk_item_seq': self.u_clk_item_seq}, [u_clk_item_seq_embedding])

            # print(self.u_clk_item_seq_batch_emb, seq_len)

            u_clk_cate_seq_hash_column = tf.contrib.feature_column.sequence_categorical_column_with_hash_bucket(
                'u_clk_cate_seq',
                self.c_hash_size,
                dtype=tf.int32
            )
            u_clk_cate_seq_embedding = tf.feature_column.embedding_column(u_clk_cate_seq_hash_column,
                                                                          self.emb_size,
                                                                          trainable=True)
            self.u_clk_cate_seq_batch_emb, _ = tf.contrib.feature_column.sequence_input_layer(
                {'u_clk_cate_seq': self.u_clk_cate_seq}, [u_clk_cate_seq_embedding])

            if self.use_negsampling:
                u_noclk_item_seq_hash_column = tf.contrib.feature_column.sequence_categorical_column_with_hash_bucket(
                    'u_noclk_item_seq',
                    self.i_hash_size,
                    dtype=tf.int32
                )
                u_noclk_item_seq_embedding = tf.feature_column.embedding_column(u_noclk_item_seq_hash_column,
                                                                                self.emb_size,
                                                                                trainable=True)
                self.u_noclk_item_seq_batch_emb, _ = tf.contrib.feature_column.sequence_input_layer(
                    {'u_noclk_item_seq':  self.u_noclk_item_seq}, [u_noclk_item_seq_embedding]
                )

                u_noclk_cate_seq_hash_column = tf.contrib.feature_column.sequence_categorical_column_with_hash_bucket(
                    'u_noclk_cate_seq',
                    self.c_hash_size,
                    dtype=tf.int32
                )
                u_noclk_cate_seq_embedding = tf.feature_column.embedding_column(u_noclk_cate_seq_hash_column,
                                                                                self.emb_size,
                                                                                trainable=True)
                self.u_noclk_cate_seq_batch_emb, _ = tf.contrib.feature_column.sequence_input_layer(
                    {'u_noclk_cate_seq': self.u_noclk_cate_seq}, [u_noclk_cate_seq_embedding]
                )

        self.item_emb = tf.concat([self.item_batch_emb, self.cate_batch_emb], 1)  # [None, emb_size * 2]
        self.u_clk_item_his_emb = tf.concat([self.u_clk_item_seq_batch_emb, self.u_clk_cate_seq_batch_emb], 2)   # [None,None,emb_size * 2]
        self.u_clk_item_his_emb_sum = tf.reduce_sum(self.u_clk_item_his_emb, 1) # [None, emb_size * 2]

        if self.use_negsampling:
            self.u_noclk_item_his_emb = tf.concat([self.u_noclk_item_seq_batch_emb, self.u_noclk_cate_seq_batch_emb], 2)
            self.u_noclk_item_his_emb_sum = tf.reduce_sum(self.u_noclk_item_his_emb, 1)

        with tf.variable_scope("var/rnn_layer_one", reuse=tf.AUTO_REUSE):
            rnn_outputs, _ = dynamic_rnn(GRUCell(self.hidden_size),
                                         self.u_clk_item_his_emb, sequence_length=self.item_seq_len,
                                         dtype=tf.float32, scope='gru1')

        # aux_loss_one = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.u_clk_item_his_emb[:, 1:, :],
        #                                    self.u_noclk_item_his_emb[:, 1:, :], self.mask[:, 1:], stag="gru")
        #
        # self.aux_loss = aux_loss_one

        with tf.variable_scope("var/attention_layer_1", reuse=tf.AUTO_REUSE):
            att_outputs, alphas = din_fcn_attention(self.item_emb, rnn_outputs, self.attention_size, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(self.hidden_size), inputs=rnn_outputs,
                                                     att_scores=tf.expand_dims(alphas, -1),
                                                     sequence_length=self.item_seq_len, dtype=tf.float32,
                                                     scope="gru2"
                                                     )

        outputs = tf.concat([self.uid_batch_emb, self.item_emb, self.u_clk_item_his_emb_sum,
                             self.item_emb * self.u_clk_item_his_emb_sum,
                             self.u_noclk_item_his_emb_sum, final_state2], 1)

        return self.build_fcn_net(outputs, use_dice=True)[:, 1]

    def build_fcn_net(self, input, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=input, name='bn1')
        dnn1 = tf.layers.dense(bn1, 128, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, scope='prelu1')

        dnn2 = tf.layers.dense(dnn1, 64, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, scope='prelu2')

        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        return tf.nn.softmax(dnn3) + 0.00000001

    def auxiliary_net(self, input, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=input, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 64, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 32, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        output = tf.nn.softmax(dnn3) + 0.00000001
        return output

    def auxiliary_loss(self, h_states, clk_seq, noclk_seq, mask, stag=None):
        mask = tf.cast(mask, tf.float32)
        clk_input = tf.concat([h_states, clk_seq], -1)
        noclk_input = tf.concat([h_states, noclk_seq], -1)
        clk_prop_ = self.auxiliary_net(clk_input, stag=stag)[:, :, 0]
        noclk_prop_ = self.auxiliary_net(noclk_input, stag=stag)[:, :, 0]
        clk_loss_ = -tf.reshape(tf.log(clk_prop_), [-1, tf.shape(clk_seq)[1]]) * mask
        noclk_loss_ = - tf.reshape(tf.log(1.0 - noclk_prop_), [-1, tf.shape(noclk_seq)[1]]) * mask
        loss_ = tf.reduce_mean(clk_loss_ + noclk_loss_)
        return loss_

    def cal_loss(self):
        sample_loss = focal_loss(
            self.label,
            self.pred,
            gamma=self.fl_gamma,
            alpha=self.fl_alpha
        )
        if self.sample_weight is not None:
            sample_loss *= self.sample_weight
        loss = tf.reduce_mean(sample_loss)
        # if self.use_negsampling:
        #     loss += self.aux_loss
        return loss

    def build_model(self, uid, item_id, cate_id, u_clk_item_seq, u_clk_cate_seq,
                    u_mask_seq=None, u_noclk_item_seq=None, u_noclk_cate_seq=None,
                    label=None, w=None, mode=None, hooks=None):
        self.mode = mode
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
            self.uid = uid
            self.item_id = item_id
            self.cate_id = cate_id
            self.u_clk_item_seq = u_clk_item_seq
            self.u_clk_cate_seq = u_clk_cate_seq
            self.mask = u_mask_seq

            ## use neg sample
            print("use neg sampling {}".format(self.use_negsampling))
            if self.use_negsampling:
                self.u_noclk_item_seq = u_noclk_item_seq
                self.u_noclk_cate_seq = u_noclk_cate_seq

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
                self.loss = self.cal_loss()
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
    batch_size, emb_size = 2, 5

    uid = [[333],[111],[222]]
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
        [33445, 55566, 77788],
        [33475, 66666, 88889],
        [99009, 89056, 899889]
    ]

    u_noclk_cate_seq = [
        [3,4,5],
        [7,8,9],
        [20,21,22]
    ]

    u_mask_seq = [
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ]
    data_w = np.random.standard_normal(size=(batch_size, 1))
    data_label = np.random.random_integers(0, 1, size=(batch_size, 1))

    placeholder_uid = tf.placeholder(dtype=tf.int64, shape=(None, 1))
    placeholder_item_id = tf.placeholder(dtype=tf.int64, shape=(None, 1))
    placeholder_cate_id = tf.placeholder(dtype=tf.int64, shape=(None, 1))

    placeholder_item_seq = tf.placeholder(dtype=tf.int64, shape=(None, 3))
    placeholder_cate_seq = tf.placeholder(dtype=tf.int64, shape=(None, 3))
    placeholder_mask_seq = tf.placeholder(dtype=tf.float32, shape=(None, 3))
    placeholder_label = tf.placeholder(dtype=tf.int64, shape=(None, 1))
    placeholder_w = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    placeholder_no_item_seq = tf.placeholder(dtype=tf.int64, shape=(None, 3))
    placeholder_no_cate_seq = tf.placeholder(dtype=tf.int64, shape=(None, 3))




    model = DeepInterestEvolutionNet(
                                    emb_size=emb_size,
                                    hidden_size=10,
                                    attention_size=10,
                                    u_seq_len=3,
                                    u_hash_size=10,
                                    i_hash_size=5,
                                    c_hash_size=3,
                                    fl_gamma=0,
                                    fl_alpha=1.0,
                                    lr=0.01,
                                    batch_norm=True,
                                    use_sample_weight=True,
                                    use_negsampling=True
                                )

    model.build_model(uid=placeholder_uid,
                      item_id=placeholder_item_id,
                      cate_id=placeholder_cate_id,
                      u_clk_item_seq=placeholder_item_seq,
                      u_clk_cate_seq=placeholder_cate_seq,
                      u_mask_seq=placeholder_mask_seq,
                      u_noclk_item_seq=placeholder_no_item_seq,
                      u_noclk_cate_seq=placeholder_no_cate_seq,
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
                                   model.uid: uid,
                                   model.item_id: pgc_id,
                                   model.cate_id: cate_id,
                                   model.u_clk_item_seq: u_clk_item_seq,
                                   model.u_clk_cate_seq: u_clk_cate_seq,
                                   model.mask: u_mask_seq,
                                   model.u_noclk_item_seq: u_noclk_item_seq,
                                   model.u_noclk_cate_seq: u_noclk_cate_seq,
                                   model.label: data_label,
                                   model.sample_weight: data_w
                               })

            print(loss)
        print(sess.run(model.pred, feed_dict={
            model.uid: uid,
            model.item_id: pgc_id,
            model.cate_id: cate_id,
            model.u_clk_item_seq: u_clk_item_seq,
            model.u_clk_cate_seq: u_clk_cate_seq,
            model.mask: u_mask_seq,
            model.u_noclk_item_seq: u_noclk_item_seq,
            model.u_noclk_cate_seq: u_noclk_cate_seq,
        }))


