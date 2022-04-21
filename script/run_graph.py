
# %%
# Mapper example code
import matplotlib.pyplot as pl
import networkx as nx
import numpy as np
import importlib
from topolearn.graph import Mapper
from topolearn.util import cluster
from topolearn.util import plot_graph_with_data
from sklearn.datasets import make_moons, make_circles


X1, _ = make_circles(noise=0.1, random_state=1, n_samples=1000)
X2, _ = make_moons(noise=0.1, random_state=1, n_samples=1000)
X2[:,0] += 2
X = np.vstack([X1,X2])

X = X1

learner = Mapper(n_intervals = 20)
graph = learner.fit(X)
plot_graph_with_data(graph, X, alpha=1)

# learner.transform(X1)

#Xr = np.flip(X, axis=1) 
#learner = Mapper(n_intervals = 20)
#graph = learner.fit(Xr)

#plot_graph_with_data(graph, Xr, alpha=1)




# %%
# Growing Neural Gas example code
import matplotlib.pyplot as pl
import numpy as np
import networkx as nx
from topolearn.graph import gng
from topolearn.util import plot_graph_with_data
from sklearn.metrics import mean_squared_error
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

reconstructed = learner.transform(X)

mse = mean_squared_error(X, reconstructed)


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
