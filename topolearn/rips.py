import numpy as np
import networkx as nx
from itertools import combinations

# Vietoris-Rips filtering
class RipsComplex:
    def __init__(self, debug=1):
        self.debug = debug

    # The Vietoris-Rips complex is calculated from the distances alone
    def fit(self, X):
        X_dist = calc_distance_matrix(X)
        return self.fit_distances(X_dist)

    def fit_distances(self, X_dist, max_dim=2, max_radius=None, max_simplices = None,  num_steps=10):

        X_dist_lower = np.tril(X_dist)
        if max_radius is None:
            max_radius = np.max(X_dist)
        # Linear breaks for now. Try area/volumebased for finer resolution?
        # Or unique, sorted distances?
        breaks = np.linspace(0, max_radius, num=num_steps, endpoint=True)
        #breaks = np.sort(np.unique(X_dist_lower))

        # Simplex added counter (index to boundary matrix)
        sidx = 0
        # Keep an index of all the added simplices, index and filtration value
        simplex_collection = {}
        # Add the points as 0-simplices
        for i in range(0, len(X_dist)):
            # Value of a simlex is (index, distance, dimension)
            simplex = frozenset([i])
            simplex_collection[simplex] = (i, 0, 0)
            #simplices[0].append(simplex)
            sidx += 1

        # Pass 1 - add 1 simplices5
        eps_prev = 0
        for t, eps in enumerate(breaks):
            # Find all new edges
            within = np.where((X_dist_lower > eps_prev) & (X_dist_lower <= eps))
            # If no new points are within range, skip to next eps
            if len(within[0]) == 0:
                continue
            edges = [frozenset({i, j}) for i, j in np.transpose(within)]
            # simplices[1].extend(edges)
            for edge in edges:
                simplex_collection[edge] = (sidx, 1, eps)
                sidx += 1
            # Find the higher order simplices
            simplices_added_prev_dim = edges  # Simplices added lower dimension
            simplices_new = []  # Simplices added current dimension
            for dim in range(1, max_dim):
                for simplex in simplices_added_prev_dim:
                    # For current distance, check if any new nodes have reached
                    # epsilon-distance, and add these to d+1 dimensional simplices
                    point_dist = np.max(X_dist[:, tuple(simplex)], axis=1)
                    # within = np.where((point_dist > eps_prev) & (point_dist <= eps))
                    within = np.where(point_dist <= eps)
                    # Ignore points already in simplex
                    point_set = frozenset(within[0]) - simplex
                    if len(point_set) == 0:
                        continue
                    # Add each new point to the simplex
                    for point in point_set:
                        # New simplex is union of new point and old points
                        new_simplex = simplex | frozenset({point})
                        # Avoid counting the same simplex more than once for the same filter value
                        # (matters only for counter, simplices are unique)
                        if not new_simplex in simplex_collection:
                            simplices_new.append(new_simplex)
                            simplex_collection[new_simplex] = (sidx, dim, eps)
                            sidx += 1
                simplices_added_prev_dim = simplices_new  
            eps_prev = eps
            if self.debug:
                print(f"eps={eps}, n={len(simplex_collection)}")
            if max_simplices is not None and len(simplex_collection) > max_simplices:
                print("Reached max number of simplices, stopping")
                break

        self.simplical_complex = SimplicalComplex(simplex_collection)
        return self.simplical_complex
        

# The Vietoris-Rips complex can be calculated from distances alone,
# which both simplifies calcualations, and make it possible to apply
# the filtering on other distances than euclidan.
# Create a distance matrix from input X feature matrix.
def calc_distance_matrix(X):
    dist_matrix = np.zeros((X.shape[0], X.shape[0]))
    for j, xj in enumerate(X):
        dist_matrix[j, :] = np.linalg.norm(X - X[j, :], axis=1)
    return dist_matrix


# Container class for the simplical complexes
# Init with a simplex_collection dictionary: 
#    keys: frozenset({nodes})
#    values: tuple(counter, added_idx, filtration_value )
class SimplicalComplex:

    def __init__(self, simplex_collection):
        self.simplex_collection = simplex_collection

    # Return the 1-skeleton of the simplex as a
    # networkx graph, add coordinates of X is given.
    def graph(self, X = None):
        graph = nx.Graph()
        w = None
        for simplex_set, (idx, dim, fvalue) in self.simplex_collection.items():
            simplex = tuple(simplex_set) # Make subscribtable 
            if dim == 0:
                if X is not None:
                    w = X[simplex[0]]
                graph.add_node(simplex[0], w=w,i=idx,f=fvalue )
            elif dim ==1:
                graph.add_edge(simplex[0], simplex[1])

        return graph       

    # Return list indexed by index-number
    def as_list(self):
        simplex_list = [ [ dim, fvalue, simplex, idx ] for  simplex, (idx, dim, fvalue) in  self.simplex_collection.items() ] 
        return simplex_list

    def boundary_matrix(self):
        size = len(self.simplex_collection)
        bmatrix = np.zeros((size, size), dtype=bool)
        for simplex_set, (idx, dim, fvalue) in self.simplex_collection.items():
            if dim >= 1: 
                for boundary in combinations(simplex_set, dim):
                    # Get the index number of the faces of the simplex
                    face = frozenset(boundary)
                    bmatrix[self.simplex_collection[face][0], idx] = True
        return bmatrix

        

# Reduce the boundary matrix
def reduce_matrix(boundary_matrix):
    # Passed by ref; make a copy
    reduced_matrix = boundary_matrix.copy()
    dim = reduced_matrix.shape[0]
    v_matrix = np.eye(dim, dtype=bool)
    # Initial low-values for matrix. For reduced columns, we set
    # low(B_i) = -1, otherwise low(B_i) = max_j{j: B_ij != 0} 
    low = [ np.max(np.where(col), initial=-1) for col in reduced_matrix.T ]
    # Main algorithm
    for j in range(0, dim):
        while True:
            if low[j] == -1: # Col fully reduced
                break
            [cols] = np.where(low[0:j]==low[j])
            if len(cols) == 0:
                break        # No columns left to add
            i = cols[0]
            # Add the columns mod 2 
            reduced_matrix[:,j] = np.logical_xor(reduced_matrix[:,j], reduced_matrix[:,i])
            v_matrix[:,j] = np.logical_xor( v_matrix[:,j], v_matrix[:,i])
            # Update the low function 
            low[j] = np.max(np.where(reduced_matrix[:,j]), initial=-1) 
    return (reduced_matrix, v_matrix)

# Given the reduced matrix, return the birth-death pairs.
# Returns a list ( )
# 
def birth_death_pairs(reduced_matrix):
    low = np.array([ np.max(np.where(col), initial=-1) for col in reduced_matrix.T ])
    birth_death_pair = list()
    for j,i in enumerate(low):
        if i == -1:   # Birth
            [col] = np.where(low==j)
            if len(col) == 0:   # No death, None for infinity
                birth_death_pair.append((j, None))
            else:  
                birth_death_pair.append((j, col[0]))
    return birth_death_pair
















1

