
# %%
# Plotpairs
import matplotlib.pyplot as pl
import numpy as np
import networkx as nx
from topolearn import persistence as ph
from topolearn.simpcomplex import RipsComplex, distance_matrix
from topolearn import util

import importlib
importlib.reload(ph)
importlib.reload(simpcomplex)

from sklearn.datasets import make_moons, make_circles
X1, y1 = make_circles(noise=0.2,  n_samples=100, random_state=50)
X2, y2 = make_circles(noise=0.2,  n_samples=50, random_state=50)
X2 = X2 * 0.5
X2[:, 0] += 2
X = np.array(np.concatenate((X1, X2), axis=0))

learner = RipsComplex(max_simplices=50000, max_dim = 2)
learner.debug_test = False
X_dist = distance_matrix(X)

simplices = learner.fit(X_dist)

util.plot_graph_with_data(simplices.graph(X), X)

pairs = simplices.birth_death_pairs()

util.plot_persistance_diagram(pairs, max_dim=1)
graph = simplices.graph(X)

plant = ph.PersistenceLandscape()
mat = plant.fit(pairs, resolution=400)
pl.figure()
for row in mat:
    pl.plot(plant.grid_m, row)

pimg = ph.PersistenceImage()
pimg.fit(pairs, sigma=0.1, resolution=50)
pimg.plot()



#%%

# To the same with the alpha complex. Discovers the same topology in 
# a fraction of the time.

learner = simpcomplex.AlphaComplex()

simplices = learner.fit(X)
util.plot_graph_with_data(simplices.graph(X), X)

pairs = simplices.birth_death_pairs()

util.plot_persistance_diagram(pairs, max_dim=1)
graph = simplices.graph(X)

pland = ph.PersistenceLandscape()
mat = pland.fit(pairs, resolution=400)
pl.figure()
for row in mat:
    pl.plot(plant.grid_m, row)

pimg = pimage.PersistenceImage()
pimg.fit(pairs, sigma=0.1, resolution=50)
pimg.plot()



#pl.figure()
#pl.imshow(pimg.images[0], origin='lower', extent= pimg.extent,  cmap='Blues')
##pl.figure()
#p#l.imshow(pimg.images[1], origin='lower', extent= pimg.extent,  cmap='Blues')



# %%

# todo- test with the

y, X = util.make_shells(400, 3, noise=0)
X_dist = simpcomplex.distance_matrix(X)

# learner = rips.RipsComplex( max_simplices=10000)


learner = simpcomplex.AlphaComplex()

simplices = learner.fit(X)
util.plot_graph_with_data(simplices.graph(X), X, axis=True)

graph = simplices.graph(X)

bdpairs = simplices.birth_death_pairs()
bmatrix = simplices.boundary_matrix()

util.plot_persistance_diagram(bdpairs)

pland = ph.PersistenceLandscape()
mat = pland.fit(bdpairs, resolution=400)
pland.plot()


pimg = ph.PersistenceImage()
pimg.fit(pairs, sigma=0.1, resolution=50)

pl.figure()
pl.imshow(pimg.images[0], origin='lower', extent= pimg.extent,  cmap='Blues')
pl.figure()
pl.imshow(pimg.images[1], origin='lower', extent= pimg.extent,  cmap='Blues')





# %%
