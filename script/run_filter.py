
# %%
# Rips 
import matplotlib.pyplot as pl
import numpy as np
import networkx as nx

from topolearn import graph
from topolearn import simpcomplex
from topolearn import homology as ph
from topolearn import util
import importlib
importlib.reload(graph)
importlib.reload(ph)
importlib.reload(simpcomplex)
importlib.reload(util)


from sklearn.datasets import make_moons, make_circles
X1, y1 = make_circles(noise=0.05,  n_samples=200, random_state=50)
X2, y2 = make_circles(noise=0.05,  n_samples=100, random_state=50)
X2 = X2 * 0.5
X2[:, 0] += 2
X = np.array(np.concatenate((X1, X2), axis=0))

learner = simpcomplex.RipsComplex(max_simplices = 50000, max_dim = 4, num_steps=1000)
X_dist = simpcomplex.distance_matrix(X)

simplices = learner.fit(X_dist)
graph = simplices.graph(X)

bdpairs = simplices.birth_death_pairs()

util.plot_graph_with_data(graph, X)
util.plot_persistance_diagram(bdpairs, max_dim=1)


# bmatrix = simplices.boundary_matrix()

#rmatrix = simpcomplex.reduce_matrix(bmatrix)
#rmatrix_2 = simpcomplex.reduce_matrix_bit(bmatrix)
#homologypairs = simplices.birth_death_pairs()

#pairs = np.array(homologypairs )

#sc = simplices.simplex_collection





# %%
import matplotlib.pyplot as pl
from sklearn.datasets import make_moons, make_circles
import numpy as np
import networkx as nx
from topolearn import simpcomplex
from topolearn import homology as ph
from topolearn import util
import importlib
importlib.reload(ph)
importlib.reload(util)


X1, y1 = make_circles(noise=0.125,  n_samples=80, random_state=50)
X2, y2 = make_circles(noise=0.125,  n_samples=30, random_state=50)
X2 = X2 * 0.5
X2[:, 0] += 2
X = np.array(np.concatenate((X1, X2), axis=0))

learner = simpcomplex.AlphaComplex()
simplices = learner.fit(X)
graph = simplices.graph(X)

bdpairs = simplices.birth_death_pairs()

util.plot_graph_with_data(graph, X, axis=True)
util.plot_persistance_diagram(bdpairs)


# %%

import matplotlib.pyplot as pl
from sklearn.datasets import make_moons, make_circles
import numpy as np
import networkx as nx
from topolearn import simpcomplex
from topolearn import homology
from topolearn import util
import importlib
importlib.reload(homology)
importlib.reload(simpcomplex)
importlib.reload(util)


# X,y = make_circles(noise=0.0001,  n_samples=4, random_state=50)

X = np.array([[0,0],[1,0],[0,1],[1,1]])


l1 = simpcomplex.AlphaComplex()
s1 = l1.fit(X)
graph = s1.graph(X)


bdpairs = s1.birth_death_pairs()
bmatrix = s1.boundary_matrix()
rmatrix = homology.reduce_matrix(bmatrix)

util.plot_graph_with_data(graph, X)
util.plot_persistance_diagram(bdpairs)

X_dist = rips.calc_distance_matrix(X)
l2 =  rips.RipsComplex()
s2 = l2.fit(X_dist)
bd2 = s2.birth_death_pairs()





# %%
# Rips 
import matplotlib.pyplot as pl
import numpy as np
import networkx as nx

from topolearn import graph
from topolearn import simpcomplex
from topolearn import homology as ph
from topolearn import util
import importlib
importlib.reload(graph)
importlib.reload(ph)
importlib.reload(simpcomplex)
importlib.reload(util)


from sklearn.datasets import make_moons, make_circles
X1, y1 = make_circles(noise=0.05,  n_samples=100, random_state=50)
X2, y2 = make_circles(noise=0.05,  n_samples=50, random_state=50)
X2 = X2 * 0.5
X2[:, 0] += 2
X = np.array(np.concatenate((X1, X2), axis=0))

learner = simpcomplex.RipsComplex(max_simplices = 100000, max_dim = 3, num_steps=1000)
X_dist = simpcomplex.distance_matrix(X)

simplices = learner.fit(X_dist)
graph = simplices.graph(X)

bdpairs = simplices.birth_death_pairs()

util.plot_graph_with_data(graph, X)
util.plot_persistance_diagram(bdpairs, max_dim=1)
# %%
