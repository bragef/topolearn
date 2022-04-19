# %%
# Mapper example code
from matplotlib.pyplot import plot, scatter
import networkx as nx
import importlib

from sklearn.datasets import make_moons, make_circles

X, y = make_moons(noise=0.01, random_state=1, n_samples=100)
from topolearn.graph import mapper

importlib.reload(mapper)

M = mapper.Mapper()

intervals = M.split_intervals(X, n_intervals=10)
clusters = M.find_clusters(X, intervals)
graph = M.connect_clusters(clusters)

nx.draw_networkx(graph)


# %%
# Growing Neural Gas example code
import matplotlib.pyplot as pl
import numpy as np
import networkx as nx
from topolearn.graph import gng
from topolearn.util import plot_graph_with_data
import importlib

importlib.reload(gng)

from sklearn.datasets import make_moons, make_circles

X1, y = make_moons(noise=0.05, n_samples=2000)
X2, y = make_circles(noise=0.05, n_samples=1000)
X2[:, 0] += 2
X = np.array(np.concatenate((X1, X2), axis=0))

learner = gng.GrowingNeuralGas(max_nodes=200)
graph = learner.fit(X)

plot_graph_with_data(graph, X)

# %%
# Self Organising Map example code
import networkx as nx
from topolearn.graph import som
from topolearn.util import plot_graph_with_data

from sklearn.datasets import make_moons, make_circles

X, y = make_circles(noise=0.05, n_samples=500)

# Example 1: Fit a 5 x 5 hexagonal grid
geometry = nx.hexagonal_lattice_graph(5, 5) 
learner = som.SelfOrganisingMaps(graph=geometry, alpha=0.001, conv_rate=0.001)
graph = learner.fit(X)
plot_graph_with_data(graph, X)


# Example 2: Fit a circle
geometry = nx.cycle_graph(20)
learner = som.SelfOrganisingMaps(graph=geometry, alpha=0.001, conv_rate=0.001)
graph = learner.fit(X)
plot_graph_with_data(graph, X)



# %%
# Generative Gaussian Graph example code

from topolearn.graph import ggg
from topolearn.util import plot_graph_with_data
from sklearn.datasets import make_moons, make_circles
import numpy as np

X, y = make_moons(noise=0.05, n_samples=10000)
learner = ggg.GenerativeGaussianGraph(k=20, sigma=1, max_iter=100, init_method="KMeans")
graph = learner.fit(X)
# Plot the full graph fitted by gnn or kmeans
plot_graph_with_data(learner.graph, X)
# Plot number of eges vs log probabilties to identify interesting scales
learner.kneeplot()
# Get a pruned graph and plot
plot_graph_with_data( learner.pruned_graph(np.exp(-4)), X)



# %%
