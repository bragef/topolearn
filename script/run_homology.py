# %%
from  topolearn import simpcomplex, homology
import numpy as np
import importlib
importlib.reload(simpcomplex)
importlib.reload(homology)



D = np.array([
    [0,0,0,0,1,1,0,0,1,0,0],
    [0,0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,1,0,0,1,0,0,0],
    [0,0,0,0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,1,1],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0]
], dtype=bool)
R =  homology.reduce_matrix(D)
R2 =  homology.reduce_matrix_bit(D)
R3 =  homology.reduce_matrix_twist(D)
R4 =  homology.reduce_matrix_bit_s(D)


print(1*(R2 == R3))


# simpcomplex.birth_death_pairs(R)



# %%
from topolearn import simpcomplex, homology, rips
import numpy as np
import importlib
from time import time
importlib.reload(simpcomplex)
importlib.reload(homology)
importlib.reload(rips)
from sklearn.datasets import make_moons, make_circles


X1, y1 = make_circles(noise=0.125,  n_samples=300, random_state=50)
X2, y2 = make_circles(noise=0.125,  n_samples=202, random_state=50)
X2 = X2 * 0.5
X2[:, 0] += 2
X = np.array(np.concatenate((X1, X2), axis=0))


learner = rips.RipsComplex( max_dim = 2, num_steps = 1000, max_simplices = 100000)
X_dist = rips.calc_distance_matrix(X)
simplices = learner.fit(X_dist)

# bmatrix = simplices.boundary_matrix(sparse_mat=False)
bmatrix_sets = simplices.boundary_sets()


#  bmatrix_s = simplices.boundary_matrix_sparse()

#t = time()
#bd3m = homology.reduce_matrix_bit(bmatrix)
##bd3 = homology.find_birth_death_pairs(bd3m)
#p#rint(f'Binary : {time()-t}')

t=time()
bd5s = homology.reduce_matrix_set(bmatrix_sets)
bd5 = homology.find_birth_death_pairs_set(bd5s)
print(f'Set : {time()-t}')

bd5 == bd3

#t = time()
#bd4 = homology.reduce_matrix_bit2(bmatrix)
#print(f'Dev : {time()-t}')

#t = time()
#bd5 = homology.reduce_matrix(bmatrix)
#print(f'Plain : {time()-t}')

#t = time()
#bd2 = homology.reduce_matrix_bit_s(bmatrix)
#print(f'Binary - np.where : {time()-t}')

# import cProfile

#cProfile.run('homology.reduce_matrix_bit(bmatrix)')
#cProfile.run('homology.reduce_matrix_bit_s(bmatrix)')



# %%