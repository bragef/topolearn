# %%
# Rips 
import matplotlib.pyplot as pl
import numpy as np
import networkx as nx

from topolearn import simpcomplex
from topolearn import homology as ph
from topolearn import util
import importlib
importlib.reload(ph)
importlib.reload(simpcomplex)
importlib.reload(util)



from sklearn.datasets import make_moons, make_circles
X1, y1 = make_circles(noise=0.05,  n_samples=100, random_state=50)
X2, y2 = make_circles(noise=0.05,  n_samples=50, random_state=50)
X2 = X2 * 0.5
X2[:, 0] += 2
X = np.array(np.concatenate((X1, X2), axis=0))

learner = simpcomplex.RipsComplex(max_simplices = 1000, max_dim = 2, num_steps=1000)
X_dist = simpcomplex.distance_matrix(X)

# test = learner.fit_distances(X_dist, max_radius = , max_dim = 3)
#test = learner.fit_distances(X_dist, max_radius = 0.25, max_dim = 2, num_steps=50)
simplices = learner.fit(X_dist)
graph = simplices.graph(X)

bdpairs = simplices.birth_death_pairs()

util.plot_graph_with_data(graph, X)
util.plot_persistance_diagram(bdpairs, max_dim=1)

#%%
import numpy as np
# from sklearn.neighbors import KDTree
from scipy.spatial import KDTree
from topolearn import simpcomplex
from topolearn import homology as ph
from topolearn import util
from sklearn.datasets import make_moons, make_circles, make_blobs


from sklearn.datasets import make_moons, make_circles
X1, y1 = make_circles(noise=0.1,  n_samples=300, random_state=50)
X2, y2 = make_circles(noise=0.01,  n_samples=100, random_state=50)
X2 = X2 * 0.5
X = np.array(np.concatenate((X1, X2), axis=0))
moons, _ = make_moons(n_samples=50, noise=0.05)
blobs, _ = make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
X = np.vstack([moons, blobs])


X_dist = simpcomplex.distance_matrix(X)
X_dist_mreach =  simpcomplex.distance_matrix_mreach(X, 10)

learner = simpcomplex.RipsComplex( 
    max_dim = 2, num_steps=1000, max_simplices=1000)

simplices = learner.fit(X_dist_mreach)
graph = simplices.graph(X)
util.plot_graph_with_data(graph, X)

simplices = learner.fit(X_dist)
graph = simplices.graph(X)
util.plot_graph_with_data(graph, X)


learner = simpcomplex.AlphaComplex(max_radius=0.4)
simplices = learner.fit(X)
graph = simplices.graph(X)
util.plot_graph_with_data(graph, X)

simplices = learner.fit(X, X_dist = X_dist_mreach)
graph = simplices.graph(X)
util.plot_graph_with_data(graph, X)


# %%
