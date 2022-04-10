import numpy as np
import networkx as nx
from .simpcomplex import SimplicalComplex

# Vietoris-Rips filtering
class RipsComplex:
    def __init__(self, debug=1):
        self.debug = debug

    # Fit from distance matrix
    # (Will not work with NaNs, prefiltered values should be set to -1)
    def fit(self, X_dist, max_dim=2, max_radius=None, max_simplices = None,  num_steps=500):
        X_dist_lower = np.tril(X_dist)
        if max_radius is None:
            max_radius = np.max(X_dist)
        # Linear breaks for now. Try area/volumebased for finer resolution?
        # Or unique, sorted distances?
        breaks = np.linspace(0, max_radius, num=num_steps, endpoint=True)
        # breaks = np.sort(np.unique(X_dist_lower))

        # Simplex added counter (index to boundary matrix)
        sidx = 0
        # Keep an index of all the added simplices, index and filtration value
        simplex_collection = {}
        # Add the points as 0-simplices
        for i in range(0, len(X_dist)):
            # Value of a simlex is (index, dimension, filter distance, actual distance)
            simplex = frozenset([i])
            simplex_collection[simplex] = (i, 0, 0, 0)
            sidx += 1

        eps_prev = 0
        for t, eps in enumerate(breaks):
            # Find all new edges
            within = np.where((X_dist_lower > eps_prev) & (X_dist_lower <= eps))
            # If no new points are within range, skip to next filtration value
            if len(within[0]) == 0:
                continue
            edges = [frozenset({i, j}) for i, j in np.transpose(within)]
            # Todo: sort the edges by distance!
            for edge in edges:
                simplex_collection[edge] = (sidx, 1, points_max_distance(X_dist, edge), eps)
                sidx += 1
            # Find the higher order simplices
            simplices_added_prev_dim = edges  # Simplices added lower dimension
            simplices_new = []  # Simplices added current dimension
            for dim in range(2, max_dim+1):
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
                        # (matters only for counter, index values ensures that simplices are unique)
                        if not new_simplex in simplex_collection:
                            simplices_new.append(new_simplex)
                            # Calculate the length of the edge wich completes the simplex to
                            # get a continous birth value for the simplex rather than the
                            # discrete values from the Rips.
                            simplex_max_dist = points_max_distance(X_dist, new_simplex)
                            simplex_collection[new_simplex] = (sidx, dim, simplex_max_dist, eps)
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

def points_max_distance(X_dist, simplex):
    return np.max(X_dist[np.ix_(tuple(simplex), tuple(simplex))])

