#%%


def plot_orig_vs_recon(title='', n_samples=3):
    fig = plt.figure(figsize=(10,6))
    plt.suptitle(title)
    for i in range(3):
        plt.subplot(3, 1, i+1)
        idx = random.sample(range(x_train.shape[0]), 1)
        plt.plot(autoencoder.predict(x_train[idx]).squeeze(), label='reconstructed' if i == 0 else '')
        plt.plot(x_train[idx].squeeze(), label='original' if i == 0 else '')
        fig.axes[i].set_xticklabels(metric_names)
        plt.xticks(np.arange(0, 10, 1))
        plt.grid(True)
        if i == 0: plt.legend();

plot_orig_vs_recon('Before training the encoder-decoder')

model_history = autoencoder.fit(x_train, x_train, epochs=5000, batch_size=32, verbose=0)

plt.plot(model_history.history["loss"])
plt.title("Loss vs. Epoch")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.grid(True)

encoded_x_train = encoder(x_train)
plt.figure(figsize=(6,6))
plt.scatter(encoded_x_train[:, 0], encoded_x_train[:, 1], alpha=.8)
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2');


model_history = autoencoder.fit(x_train, x_train, epochs=5000, batch_size=32, verbose=0)

plt.plot(model_history.history["loss"])
plt.title("Loss vs. Epoch")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.grid(True)












# %%
import matplotlib.pyplot as pl
import numpy as np
import networkx as nx
from topolearn import simpcomplex
from topolearn import homology
from topolearn import util
import importlib
importlib.reload(simpcomplex)
importlib.reload(homology)
importlib.reload(util)


#X = util.make_shells(500, 3, noise=0.001)

#learner = rips.RipsComplex()
#X_dist = rips.calc_distance_matrix(X)

#simplices = learner.fit(X_dist, max_radius=1, max_dim = 2, num_steps=50)
#graph = simplices.graph(X)

#util.plot_graph_with_data(graph, X)

# 
y, X = util.make_shells(200, 3, noise=0)
X_dist = simpcomplex.calc_distance_matrix(X)
learner = simpcomplex.RipsComplex( max_simplices=50000)

simplices = learner.fit(X_dist)
graph = simplices.graph(X)


util.plot_graph_with_data(graph, X, axis=True)

bdpairs = simplices.birth_death_pairs()
bmatrix = simplices.boundary_matrix()

util.plot_persistance_diagram(bdpairs)



# %%
import matplotlib.pyplot as pl
import numpy as np
import networkx as nx
from topolearn import rips 
from topolearn import simpcomplex
from topolearn import alphacomplex
from topolearn import homology
from topolearn import util
import importlib
importlib.reload(rips)
importlib.reload(simpcomplex)
importlib.reload(homology)
importlib.reload(util)


#X = util.make_shells(500, 3, noise=0.001)

#learner = rips.RipsComplex()
#X_dist = rips.calc_distance_matrix(X)

#simplices = learner.fit(X_dist, max_radius=1, max_dim = 3, num_steps=50)
#graph = simplices.graph(X)

#util.plot_graph_with_data(graph, X)

# 
y, X = util.make_shells(50, 2, noise=0.01)

# learner = alphacomplex.AlphaComplex()
learner = rips.RipsComplex(max_dim=2, verbose=1)
X_dist = rips.calc_distance_matrix(X)

simplices = learner.fit(X_dist)
graph = simplices.graph(X)

bdpairs = simplices.birth_death_pairs()
bmatrix = simplices.boundary_matrix()

util.plot_graph_with_data(graph, X, axis=True)
util.plot_persistance_diagram(bdpairs, max_dim=1)






# %%



