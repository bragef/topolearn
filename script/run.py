# %%
# Mapper example code
from matplotlib.pyplot import plot, scatter
import networkx as nx
import importlib

from sklearn.datasets import make_moons, make_circles

X, y = make_moons(noise=0.01, random_state=1, n_samples=100)

import topolearn.mapper as mapper

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
from topolearn import gng
import importlib

importlib.reload(gng)

from sklearn.datasets import make_moons, make_circles

X, y = make_moons(noise=0.1, n_samples=1000)

X1, y = make_moons(noise=0.05, n_samples=2000)
X2, y = make_circles(noise=0.05, n_samples=1000)
X2[:, 0] += 2
X = np.array(np.concatenate((X1, X2), axis=0))

# X, y = make_circles(noise=0.1, random_state = 1, n_samples=10000)


learner = gng.GrowingNeuralGas(max_nodes=200)
graph = learner.fit(X)

pos = {n: d["w"] for (n, d) in graph.nodes(data=True)}
ax = pl.scatter(X[:, 0], X[:, 1], s=1, alpha=0.15)
nx.draw(graph, pos, node_color="r", edge_color="b", alpha=0.5, node_size=2)

# %%
# Self Organising Map example code
import matplotlib.pyplot as pl
import numpy as np
import networkx as nx
from topolearn import som
import importlib
importlib.reload(som)

from sklearn.datasets import make_moons, make_circles


X, y = make_circles(noise=0.05, n_samples=500)

# Example 1: Fit a 5 x 5 hexagonal grid
geometry = nx.hexagonal_lattice_graph(5, 5)
learner = som.SelfOrganisingMaps(graph=geometry, alpha=0.001, conv_rate=0.001)

graph = learner.fit(X)

pl.figure()
pos = {n: d["w"] for (n, d) in graph.nodes(data=True)}
ax = pl.scatter(X[:, 0], X[:, 1], s=1, alpha=0.15)
nx.draw(graph, pos, node_color="r", edge_color="b", alpha=0.5, node_size=2)

# Example 2: Fit a circle
geometry = nx.cycle_graph(20)
learner = som.SelfOrganisingMaps(graph=geometry, alpha=0.001, conv_rate=0.001)
graph = learner.fit(X)
pl.figure()
pos = {n: d["w"] for (n, d) in graph.nodes(data=True)}
ax = pl.scatter(X[:, 0], X[:, 1], s=1, alpha=0.15)
nx.draw(graph, pos, node_color="r", edge_color="b", alpha=0.5, node_size=2)


# pl.scatter(v[:,0], v[:,1], s=10)


# %%
# Generative Gaussian Graph example code
import matplotlib.pyplot as pl
import numpy as np
import networkx as nx
from topolearn import ggg
import importlib

importlib.reload(ggg)

from sklearn.datasets import make_moons, make_circles

# X, y = make_moons(noise=0.05,  n_samples=100000)
X, y = make_moons(noise=0.05, n_samples=1000)

# X, y = make_circles(noise=0.01, random_state = 1, n_samples=10000)

# Fit
learner = ggg.GenerativeGaussianGraph(k=20, sigma=1, max_iter=100, init_method="KMeans")
graph = learner.fit(X)
edgeprobs = learner.edge_probs()

# Initial fit
pl.figure()
pos = {n: d["w"] for (n, d) in graph.nodes(data=True)}
ax = pl.scatter(X[:, 0], X[:, 1], s=1, alpha=0.25)
nx.draw(graph, pos, node_color="r", edge_color="b", alpha=0.5, node_size=2)

# Identify cutoff values using a knee plot; select -4 as value
# here
pl.figure()
pl.plot(np.log(edgeprobs), range(0, len(edgeprobs)))

pl.figure()
nx.draw(
    learner.pruned_graph(np.exp(-4)),
    pos,
    node_color="r",
    edge_color="b",
    alpha=0.5,
    node_size=2,
)


# %%
