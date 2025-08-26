from utils.modules_v2 import *

from utils.loss_func import get_loss_func
import numpy as np

class DragonNetModel(tf.keras.Model):

    def __init__(self,
                 x_dim,
                 sparse_x_len,
                 emb_size,
                 layer_units,
                 layer_activations=None,
                 tower_units=None,
                 tower_activations=None,
                 batch_norm=False):

        super(DragonNetModel, self).__init__()
        self.x_dim = x_dim
        self.emb_size = emb_size
        self.sparse_x_len = sparse_x_len
        self._embedding = EmbeddingLookup(name='fea_emb',
                                          vocab_size=self.x_dim,
                                          embedding_size=self.emb_size,
                                          initializer_range=0.01,
                                          embedding_name='id_embedding')

        if type(layer_units) is not list:
            layer_units = [int(x) for x in layer_units.split(',')]

        if type(layer_activations) is not list:
            layer_activations = list(map(lambda x: None if (x == 'None') else x, layer_activations.split(',')))

        assert len(layer_units) == len(layer_activations), 'layer units length and layer activations length not match'

        self.mlp1 = MLP(units=layer_units, activations=layer_activations, use_batch_norm=batch_norm)
        self.mlp2 = MLP(units=[1], activations=['sigmoid'], use_batch_norm=batch_norm)

        if type(tower_units) is not list:
            tower_units = [int(x) for x in tower_units.split(',')]

        if type(tower_activations) is not list:
            tower_activations = list(map(lambda x: None if (x == 'None') else x, tower_activations.split(',')))

        self.mlp_tower = MLP(units=tower_units, activations=tower_activations, use_batch_norm=batch_norm)
        self.el = EpsilonLayer()


    def call(self, inputs, training=None, mask=None):

        x = inputs['x']
        x = tf.reshape(x, shape=(-1, self.sparse_x_len))   # 任何情况下x都不能为空
        y = tf.cast(tf.reshape(inputs['y'], (-1, 2)), dtype=tf.float32)

        dis_embs = self._embedding(x)
        dense_x = tf.reshape(dis_embs, shape=(-1, self.sparse_x_len * self.emb_size))

        z = self.mlp1(dense_x)
        q0 = self.mlp_tower(z)
        q1 = self.mlp_tower(z)
        g = self.mlp2(z)

        return tf.keras.layers.Concatenate(1)([q0, q1, g, y])



if __name__ == '__main__':
    units = [64, 64, 64]
    activations = ['relu', 'relu', 'sigmoid']
    tower_units = [16, 16, 1]
    tower_activations = ['relu', 'relu', 'sigmoid']
    seq_length = 15
    seq_dim = 11
    context_dim = 8
    sparse_context_range = [1, 9]
    position_attention = False
    enable_drm = True

    # mock data
    batch_size, x_dim, sparse_x_len, emb_size = 32, 100, 20, 5
    data_x = np.random.random_integers(0, x_dim - 1, size=(batch_size, sparse_x_len))

    # data_y = [[1, 0],
    #           [1, 1],
    #           [0, 1]]
    data_y = np.random.randint(0, 2, size=(batch_size, 2))
    data_w = np.random.random_integers(1, 5, size=(batch_size, 2))


    configs = {
        'x_dim': x_dim,
        'sparse_x_len': sparse_x_len,
        'emb_size': emb_size,
        'layer_units': units,
        'layer_activations': activations,
        'tower_units': tower_units,
        'tower_activations': tower_activations,
        'loss_func': "propensity_loss",
        'lr': 0.001,
        'alpha': 1.0,
        'beta': 1.0,
        'g_weight': 0.5,
        'batch_norm': True
    }

    # model = Dragon_net(configs)

    inputs = {
        'x': tf.convert_to_tensor(data_x, dtype=tf.int32),
        'y': tf.convert_to_tensor(data_y, dtype=tf.float32)
    }

    model = DragonNetModel(
        x_dim=configs['x_dim'],
        sparse_x_len=configs['sparse_x_len'],
        emb_size=configs['emb_size'],
        layer_units=configs['layer_units'],
        layer_activations=configs['layer_activations'],
        tower_units=configs['tower_units'],
        tower_activations=configs['tower_activations'],
        batch_norm=True
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=configs['lr']),
        loss=get_loss_func(configs))

    model.fit(inputs, inputs['y'], epochs=100)

    # run_config = tf.estimator.RunConfig(keep_checkpoint_max=1,
    #                                     log_step_count_steps=10)
    # # 输入model_fn，模型保存路径
    # classifier = tf.estimator.Estimator(model_fn=model.model_fn, config=run_config)
    #
    # tf.estimator.train_and_evaluate(
    #     classifier,
    #     train_spec=tf.estimator.TrainSpec(input_fn=[inputs['tf_features'], inputs['labels']]),
    #
    #     eval_spec=tf.estimator.TrainSpec(input_fn=[inputs['tf_features'], inputs['labels']]),
    #
    # )

    # model = DragonNetModel(x_dim=configs['x_dim'],
    #                      sparse_x_len=configs['sparse_x_len'],
    #                      emb_size=configs['emb_size'],
    #                      layer_units=configs['layer_units'],
    #                      layer_activations=configs['layer_activations'],
    #                      tower_units=configs['tower_units'],
    #                      tower_activations=configs['tower_activations'],
    #                        batch_norm=configs['batch_norm'])
    #
    #
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    #     loss=get_loss_func(configs))
    #
    # model.fit(inputs['tf_features'], inputs['labels'], batch_size=batch_size, epochs=100)