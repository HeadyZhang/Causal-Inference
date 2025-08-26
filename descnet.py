import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


from utils.tf_utils import *

class DESCN:
    def __init__(self, x_dim, sparse_x_len, emb_size, layer_units=None, layer_activations=None,
                 tower_units=None, tower_activations=None, lr=0.001, g_weight=None,
                 batch_norm=False, bn_decay=0.9, alpha=1.8, beta=0.5, multi_hot_dict=None,
                 nn_init='glorot_uniform', logger=None, use_sample_weight=False,
                 fl_gamma=0., fl_alpha=1., base_model='fm', prpsy_w=0.5, escvr1_w=0.5, escvr0_w=1,
                 h1_w=0.1, h0_w=0.1, mu1hat_w=0.5, mu0hat_w=1):
        self.logger = logger

        self.x_dim = x_dim
        self.sparse_x_len = sparse_x_len
        self.emb_size = emb_size
        self.lr = lr
        self.base_model = base_model
        self.tower_units = tower_units
        self.tower_activations = tower_activations
        self.batch_norm = batch_norm
        self.batch_norm_decay = bn_decay
        self.use_sample_weight = use_sample_weight

        self.alpha = alpha
        self.beta = beta
        self.fl_alpha = fl_alpha
        self.fl_gamma = fl_gamma
        self.nn_init = nn_init
        self.mode = None

        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.tower_units = tower_units
        self.tower_activations = tower_activations

        # params
        self.prpsy_w = prpsy_w
        self.escvr1_w = escvr1_w
        self.escvr0_w = escvr0_w
        self.h1_w = h1_w
        self.h0_w = h0_w
        self.mu1hat_w = mu1hat_w
        self.mu0hat_w = mu0hat_w

        self.x = None
        self.y = None
        self.t = None
        self.y_w = None
        self.t_w = None
        self.loss = None
        self.p_prpsy = None
        self.p_estr = None
        self.p_escr = None

        self.mu0_logit = None
        self.mu1_logit = None
        self.tau_logit = None
        self.p_h1 = None
        self.p_h0 = None
        self.p_tau = None

        self.summary = None
        self.global_step = None
        self.batch_norm_layer = None
        self.train_op = None
        self.ordr1_weights = None
        self.ordr2_emb_mat = None




    def train_bn_layer(self, x, scope_bn):
        return batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, is_training=True, reuse=None,
                          trainable=True, scope=scope_bn)

    def infer_bn_layer(self, x, scope_bn):
        return batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, is_training=False, reuse=None,
                          trainable=False, scope=scope_bn)


    def share_network(self, inputs):

        if self.base_model == 'fm':
            # fm前向过程
            with tf.variable_scope('var/fm_first_order', reuse=tf.AUTO_REUSE):
                self.ordr1_weights = tf.get_variable(
                    'weight_matrix',
                    shape=[self.x_dim, 1],
                    dtype=tf.float32,
                    initializer=tf.initializers.random_normal(stddev=0.01)
                )

                first_ordr = tf.nn.embedding_lookup(self.ordr1_weights, inputs) # [None, sparse_x_len, 1]
                first_ordr = tf.reduce_sum(first_ordr, 2)
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
                dis_embs = tf.nn.embedding_lookup(self.ordr2_emb_mat, inputs)  # [None, sparse_x_len, emb_size]
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
            output_res = get_mlp_network(input_tensor=output,
                            layer_units=self.layer_units,
                            layer_activations=self.layer_activations,
                            name='share_network',
                            bn_layer=self.batch_norm_layer,
                            disable_last_bn=False
                        )

            return  output_res


            # return tf.concat(out_tensors, 1)

    def prpsy_network(self, inputs):
        p_prpsy_logit = get_mlp_network(input_tensor=inputs,
                        layer_units=self.tower_units,
                        layer_activations=self.tower_activations,
                        name='prpsy_network',
                        bn_layer=self.batch_norm_layer,
                        disable_last_bn=True
                        )

        return p_prpsy_logit

    def mu1_network(self, inputs):
        mu1_logit = get_mlp_network(input_tensor=inputs,
                                    layer_units=self.tower_units,
                                    layer_activations=self.tower_activations,
                                    name='mu1_network',
                                    bn_layer=self.batch_norm_layer,
                                    disable_last_bn=True
                                    )
        return mu1_logit

    def mu0_network(self, inputs):
        mu0_logit = get_mlp_network(input_tensor=inputs,
                                    layer_units=self.tower_units,
                                    layer_activations=self.tower_activations,
                                    name='mu1_network',
                                    bn_layer=self.batch_norm_layer,
                                    disable_last_bn=True
                                    )
        return mu0_logit

    def tau_network(self, inputs):
        tau_logit = get_mlp_network(input_tensor=inputs,
                                    layer_units=self.tower_units,
                                    layer_activations=self.tower_activations,
                                    name='tau_network',
                                    bn_layer=self.batch_norm_layer,
                                    disable_last_bn=True
                                    )
        return tau_logit

    def forward(self, is_train):
        if self.batch_norm:
            if is_train:
                self.batch_norm_layer = self.train_bn_layer
            else:
                self.batch_norm_layer = self.infer_bn_layer
        ## share network
        shared_h = self.share_network(self.x)

        prpsy_logit = self.prpsy_network(shared_h)

        # logit for mu1, mu0
        mu1_logit = self.mu1_network(shared_h)
        mu0_logit = self.mu0_network(shared_h)

        # pseudo tau 计算loss
        tau_logit = self.tau_network(shared_h)

        p_h1 = tf.sigmoid(mu1_logit)
        p_h0 = tf.sigmoid(mu0_logit)

        p_tau = p_h1 - p_h0

        p_prpsy = tf.sigmoid(prpsy_logit)

        p_estr = tf.multiply(p_prpsy, p_h1)
        p_i_prpsy = 1 - p_prpsy
        p_escr = tf.multiply(p_i_prpsy, p_h0)

        return p_estr, p_escr, tau_logit, mu1_logit, mu0_logit, p_prpsy, p_h1, p_h0, p_tau


    def cal_loss(self):
        self.y_outcome, self.y_treatment = tf.reshape(self.y[:, 0], (-1, 1)), tf.reshape(self.y[:, 1], (-1, 1))
        if self.use_sample_weight:
            w_outcome, w_treatment = tf.reshape(self.w[:, 0], (-1, 1)), tf.reshape(self.w[:, 1], (-1, 1))
        else:
            w_outcome, w_treatment = 1, 1
        # loss for propensity
        self.prpsy_loss = w_treatment * self.prpsy_w * focal_loss(self.y_treatment, self.p_prpsy)
        # loss for ESTR ESCR
        self.estr_loss = w_outcome * self.escvr1_w * focal_loss(self.y_outcome * self.y_treatment, self.p_estr)
        self.escr_loss = w_outcome * self.escvr0_w * focal_loss(self.y_outcome * (1 - self.y_treatment), self.p_escr)

        # loss for TR, CR
        self.tr_loss = self.h1_w * focal_loss(self.y_outcome, self.p_h1) * w_outcome
        self.cr_loss = self.h0_w * focal_loss(self.y_outcome, self.p_h0) * w_outcome

        # loss for cross TR: mu1_prime, cross CR: mu0_prime
        self.cross_tr_loss = self.mu1hat_w * focal_loss(self.y_outcome,
                                                        tf.sigmoid(self.mu0_logit + self.tau_logit)) * w_outcome
        self.cross_cr_loss = self.mu0hat_w * focal_loss(self.y_outcome,
                                                        tf.sigmoid(self.mu1_logit - self.tau_logit)) * w_outcome

        sample_loss = self.prpsy_loss \
                      + self.estr_loss + self.escr_loss \
                      + self.tr_loss + self.cr_loss \
                      + self.cross_tr_loss + self.cross_cr_loss

        self.loss = tf.reduce_mean(sample_loss)


    def build_model(self, x=None, y=None, w=None, mode=None, hooks=None):
        self.mode = mode
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
            self.x = tf.reshape(x, shape=(-1, self.sparse_x_len))

            if y is not None:
                self.y = tf.cast(tf.reshape(y, (-1, 2)), dtype=tf.float32)

        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.p_estr, self.p_escr, self.tau_logit, self.mu1_logit, self.mu0_logit, self.p_prpsy, self.p_h1, self.p_h0, self.p_tau = self.forward(is_train=True)
            else:
                self.p_estr, self.p_escr, self.tau_logit, self.mu1_logit, self.mu0_logit, self.p_prpsy, self.p_h1, self.p_h0, self.p_tau = self.forward(is_train=False)

            if mode == tf.estimator.ModeKeys.PREDICT:
                pass
            else:
                self.w = w
                self.cal_loss()
                self.summary = tf.summary.scalar('loss', self.loss)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
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
                    out_dict = {'prediction': tf.concat([self.p_prpsy, self.p_tau,  self.p_h1, self.p_h0], axis=1)}
                    export_outputs = {
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            tf.estimator.export.PredictOutput(out_dict)}
                    spec = tf.estimator.EstimatorSpec(mode=mode,
                                                      predictions=out_dict,
                                                      export_outputs=export_outputs)

                elif mode == tf.estimator.ModeKeys.EVAL:
                    eval_metric_ops = {
                        'prpsy_loss' : tf.metrics.mean(self.prpsy_loss),
                        'estr_loss': tf.metrics.mean(self.estr_loss),
                        'escr_loss': tf.metrics.mean(self.escr_loss)
                    }

                    eval_metric_ops['pred_tr_auc'] = tf.metrics.auc(self.y[:, 0], self.p_h1)
                    eval_metric_ops['pred_cr_auc'] = tf.metrics.auc(self.y[:, 0], self.p_h0)
                    eval_metric_ops['tr_auc'] = tf.metrics.auc(self.y[:, 1], self.p_prpsy)

                    spec = tf.estimator.EstimatorSpec(mode=mode, loss=self.loss, eval_metric_ops=eval_metric_ops)

                else:
                    raise ValueError('invalid mode: %s' % mode)
                return spec

    def print(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)


if __name__ == '__main__':
    units = [64, 64, 32]
    activations = ['relu', 'relu', 'sigmoid']
    tower_units = [16, 16, 1]
    tower_activations = ['relu', 'relu', None]

    batch_size, x_dim, sparse_x_len, emb_size = 3, 100, 10, 5
    data_x = np.random.random_integers(0, x_dim - 1, size=(batch_size, sparse_x_len))

    data_y = [[1, 0],
              [1, 1],
              [0, 1]]

    data_w = np.random.random_integers(1, 5, size=(batch_size, 2))

    placeholder_x = tf.placeholder(dtype=tf.int64, shape=(None, sparse_x_len))
    placeholder_y =  tf.placeholder(dtype=tf.int64, shape=(None, 2))

    placeholder_w = tf.placeholder(dtype=tf.float32, shape=(None, 2))


    model = DESCN(x_dim, sparse_x_len, emb_size, layer_units=units, layer_activations=activations,
                  tower_units=tower_units, tower_activations=tower_activations, lr=0.001, g_weight=.1,
                  batch_norm=False, bn_decay=0.9, alpha=1.8, beta=0.5, multi_hot_dict={},
                  nn_init='glorot_uniform', logger=None, use_sample_weight=True,
                      fl_gamma=0., fl_alpha=1.)

    model.build_model(x=placeholder_x, y=placeholder_y, w=placeholder_w,
                      mode=tf.estimator.ModeKeys.TRAIN)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for _ in range(100):
            _, loss = sess.run([model.train_op, model.loss],
                               feed_dict={
                                   model.x : data_x,
                                   model.y : data_y,
                                   model.w : data_w,
                               })

            print(loss)
        print(sess.run(model.p_prpsy, feed_dict={
            model.x: data_x,
            model.y: data_y,
            model.w: data_w,
        }))