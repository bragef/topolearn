#%%

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import importlib
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Layer, Input
from tensorflow.keras import losses, Input
from tensorflow.keras import backend as K

from topolearn.util import make_shells
from topolearn import simpcomplex
from topolearn import autoencoder as topoauto

importlib.reload(simpcomplex)
importlib.reload(topoauto)


# Dimension of the original space
input_dim = 3
# Dimension of the latent space (encoding space)
latent_dim = 2

y, X = make_shells(3000, input_dim, noise=0.01)


# Autoencoder  model
encoder = Sequential(
    [
        Input(shape=(input_dim,)),
        Dense(256, activation="elu"),
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
model = Model(inputs=input_seq, outputs=output_seq)


# Training config
epochs = 250
batch_size = 100
lambda_topo = 0.0001

#
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_mse = keras.losses.MeanSquaredError()
topoloss = topoauto.TopologicalLoss(
    filtration=simpcomplex.RipsComplex(max_dim=1, max_radius=0.8, verbose=0),
)

train_dataset = X
train_dataset = tf.data.Dataset.from_tensor_slices((X, X))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)


for epoch in range(epochs):
    print(f"Start of epoch {epoch}")
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            middle = encoder(x_batch_train, training=True)
            preds = decoder(middle, training=True)

            loss_accuracy = loss_mse(y_batch_train, preds)
            loss_topo = topoloss(x_batch_train, middle)
            loss_value = loss_accuracy + lambda_topo * loss_topo

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if step % 200 == 0:
            print(
                f"Training loss (for one batch) at step {step}: mse: {loss_accuracy}, topo: {loss_topo}, total:{loss_value}"
            )
            print(f"Seen so far: {((step + 1) * batch_size) } samples")


encoded = encoder(X)

Xaut = model.predict(X)

# Plot
plt.figure()
plt.scatter(encoded[:, 0], encoded[:, 1], c=y, cmap="rainbow", s=0.5)
plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(Xaut[:, 0], Xaut[:, 1], Xaut[:, 2], c=y, cmap="rainbow", s=0.5)
plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="rainbow", linewidth=0.5, s=0.5)

# %%
