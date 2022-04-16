# %%

# Simple autoencoder example
# Very nice example: https://ekamperi.github.io/machine%20learning/2021/01/21/encoder-decoder-model.html

import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
import importlib
#import tensorflow.keras as keras
#from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.layers import Dense, Layer
#from tensorflow.keras import losses, Input
#from tensorflow.keras import backend as K

import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Layer
from keras import losses, Input
from keras import backend as K


from topolearn.util import make_shells
from topolearn import rips
from topolearn import autoencoder as topoauto

importlib.reload(rips)
importlib.reload(topoauto)


# Dimension of the original space
input_dim = 3
# Dimension of the latent space (encoding space)
latent_dim = 2

y, X = make_shells(400, input_dim, noise=0.01)

encoder = Sequential(
    [
        Dense(256, activation="elu", input_shape=(input_dim,)),
        Dense(128, activation="elu"),
        Dense(64, activation="elu"),
        Dense(32, activation="elu"),
        Dense(latent_dim, activation="elu"),
    ]
)
decoder = Sequential(
    [
        Dense(64, activation="elu", input_shape=(latent_dim,)),
        Dense(128, activation="elu"),
        Dense(256, activation="elu"),
        Dense(input_dim, activation=None),
    ]
)

input_seq = encoder.input
output_seq = decoder(encoder.output)

autoencoder = Model(inputs=input_seq, outputs=output_seq)

mse = losses.MeanSquaredError()


topoloss = topoauto.TopologicalLoss(
    filtration=rips.RipsComplex(max_dim=2, max_simplices=1000),
)

myloss = topoloss

autoencoder.compile(optimizer="adam", loss=myloss)
autoencoder.fit(X, X, epochs=100, batch_size=100, verbose=1, shuffle=True)
encoded = encoder(X)

Xaut = autoencoder.predict(X)

plt.figure()
plt.scatter(encoded[:, 0], encoded[:, 1], c=y, cmap="rainbow", s=0.5)
plt.figure()

ax = plt.axes(projection="3d")
ax.scatter(Xaut[:, 0], Xaut[:, 1], Xaut[:, 2], c=y, cmap="rainbow", s=0.5)

plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="rainbow", linewidth=0.5, s=0.5)


#%%

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import importlib
import tensorflow.keras as keras

from topolearn.util import make_shells
from topolearn import rips
from topolearn import autoencoder as topoauto
from topolearn import simpcomplex

importlib.reload(simpcomplex)
importlib.reload(rips)
importlib.reload(topoauto)


#y, X_inp = make_shells(100, input_dim, noise=0.01)
#X_enc = encoder(X)

X_inp = X
X_enc = encoded

topoloss = topoauto.TopologicalLoss(
    filtration=rips.RipsComplex(max_dim=2, max_simplices=1000, verbose=1),
)

# edges_z = topoloss.find_critical_edges(X_enc)

L = topoloss.calculate_topo_loss(X_inp, X_enc)

Ax = rips.calc_distance_matrix(X_inp)
Az = rips.calc_distance_matrix(X_enc)
# Edge-sets
edges_z = topoloss.find_critical_edges(Az)
edges_x = topoloss.find_critical_edges(Ax)




# %%


def losstest(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    losslog.append(K.sum(y_pred_f))
    # intersection = K.sum(y_true_f * y_pred_f) +
    err = keras.losses.mse(y_true, y_pred)
    return err


# "def my_funky_loss_fn(y_true, y_pred):
#  return (keras.losses.mse(y_true, y_pred)
#        + keras.backend.max(output_seq))

# loss = mse + alpha * topoloss
# autoencoder.add_loss(mse(input_seq, output_seq))
# autoencoder.add_loss(topoloss(encoder.input, encoder.output))
# autoencoder.add_loss(loss(input_seq, output_seq))


class MyActivityRegularizer(Layer):
    def __init__(self, rate=1e-2):
        super(MyActivityRegularizer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        # We use `add_loss` to create a regularization loss
        # that depends on the inputs.
        self.add_loss(self.rate * tf.reduce_sum(tf.square(inputs)))
        return inputs


# %%
