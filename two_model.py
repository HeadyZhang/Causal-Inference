from utils.modules_v2 import *

from utils.loss_func import get_loss_func

import numpy as np

class TwoModel(tf.keras.Model):
    def __init__(self,
                 x_dim,
                 sparse_x_len,
                 emb_size,
                 layer_units,
                 layer_activations=None,
                 batch_norm=False):

        super(TwoModel, self).__init__()

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

        self.nn_layer = MLP(units=layer_units, activations=layer_activations, use_batch_norm=batch_norm)
        self.infer_layer = MLP(units=[2], activations=['sigmoid'], use_batch_norm=False)

        self.el = EpsilonLayer()


    def call(self, inputs, training=None, mask=None):
        x = inputs['x']
        y = tf.cast(tf.reshape(inputs['y'], (-1, 2)), dtype=tf.float32)
        # w = tf.cast(tf.reshape(inputs['w'], (-1, 2)), dtype=tf.float32)
        x = tf.reshape(x, shape=(-1, self.sparse_x_len))
        dis_embs = self._embedding(x)
        dense_x = tf.reshape(dis_embs, shape=(-1, self.sparse_x_len * self.emb_size))
        z = self.nn_layer(dense_x)
        logits = self.infer_layer(z)
        return tf.concat([logits, y], axis=-1)

if __name__ == '__main__':
    units = [64, 64, 64]
    activations = ['relu', 'relu', 'sigmoid']

    batch_size, x_dim, sparse_x_len, emb_size = 32, 100, 20, 5

    data_x = np.random.random_integers(0, x_dim - 1, size=(batch_size, sparse_x_len))

    data_y = np.random.randint(0, 2, size=(batch_size, 2))
    data_w = np.random.random_integers(1, 5, size=(batch_size, 2))

    configs = {
        'x_dim': x_dim,
        'sparse_x_len': sparse_x_len,
        'emb_size': emb_size,
        'layer_units': units,
        'layer_activations': activations,
        'loss_func': "log_drm",
        'lr': 0.001,
        'alpha': 1.0,
        'beta': 0.0,
        'g_weight': 0.5,
        'batch_norm': True
    }

    inputs = {
        'tf_features': tf.convert_to_tensor(data_x, dtype=tf.int32),
        'labels': tf.convert_to_tensor(data_y, dtype=tf.int32),
        'weight': tf.convert_to_tensor(data_w, dtype=tf.float32)
    }

    model = TwoModel(x_dim=configs['x_dim'],
                     sparse_x_len=configs['sparse_x_len'],
                     emb_size=configs['emb_size'],
                     layer_units=configs['layer_units'],
                     layer_activations=configs['layer_activations'],
                     batch_norm=configs['batch_norm'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=configs['lr']),
                                                     loss=get_loss_func(configs))
    model.fit({'x':inputs['tf_features'], 'y': inputs['labels'], 'w': inputs['weight']}, inputs['labels'], batch_size=batch_size, sample_weight=inputs['weight'],  epochs=100)

