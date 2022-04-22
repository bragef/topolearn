
# %%
# Plotpairs
import matplotlib.pyplot as pl
import numpy as np
import networkx as nx
from topolearn.persistence import PersistenceLandscape, PersistenceImage
from topolearn.simpcomplex import RipsComplex, distance_matrix, AlphaComplex
from topolearn.util import plot_graph_with_data, plot_persistance_diagram
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

plot_graph_with_data(simplices.graph(X), X)

pairs = simplices.birth_death_pairs()

plot_persistance_diagram(pairs, max_dim=1)
graph = simplices.graph(X)

plant = PersistenceLandscape()
mat = plant.fit(pairs, resolution=00)
pl.figure()
for row in mat:
    pl.plot(plant.grid_m, row)


X,_ = make_circles(noise=0.2,  n_samples=100, random_state=50)
learner = AlphaComplex()
learner.fit(X).tranform()


pimg = PersistenceImage(sigma=0.1, resolution=50)
pimg.fit(pairs)
pimg.plot()



#%%
from topolearn.simpcomplex import AlphaComplex
from topolearn.persistence import PersistenceLandscape
from sklearn.datasets import make_circles

X, _ = make_circles(noise=0.2,  n_samples=1000, random_state=50)
learner = AlphaComplex()

simplices = learner.fit(X)
pairs = birth_death_pairs()
p = PersistenceLandscape(resolution=400)
p.fit(pairs,dim=1)
p.plot()
print(p.matrix)


matrix = pland.fit(pairs, resolution=400)



util.plot_persistance_diagram(pairs, max_dim=1)
graph = simplices.graph(X)

pimg = pimage.PersistenceImage()
pimg.fit(pairs, sigma=0.1, resolution=50)
pimg.plot()

from sklearn.datasets import make_moons, make_circles
X1, y1 = make_circles(noise=0.2,  n_samples=100, random_state=50)

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
from topolearn.simpcomplex import AlphaComplex, distance_matrix
from topolearn.util import make_shells, plot_graph_with_data, plot_persistance_diagram
from topolearn.persistence import PersistenceImage, PersistenceLandscape


X, _ = make_shells(000, dim=3, noise=0.1)

learner = AlphaComplex()
simplices = learner.fit(X)

plot_graph_with_data(simplices.graph(X), X, alpha=0.2)

pairs = simplices.birth_death_pairs()

plot_persistance_diagram(pairs)
graph = simplices.graph(X)


plant = PersistenceLandscape()
for dim in (1,2):
    mat = plant.fit(pairs,dim=dim, resolution=400)
    plant.plot()

pimg = PersistenceImage()
pimg.fit(pairs, sigma=0.1, resolution=50)
pimg.plot()

# %%
X, _ = make_circles(noise=0.01,  n_samples=1000, random_state=50)
learner = RipsComplex(max_dim=3, max_simplices=50000, input_distance_matrix=False)
simplices = learner.fit(X)
pairs = simplices.birth_death_pairs()
p = PersistenceImage(resolution=100)
p.fit(pairs)
p.plot()


# %%
