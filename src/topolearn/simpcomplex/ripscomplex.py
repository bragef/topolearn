import numpy as np
import networkx as nx
from time import time
from .simpcomplex import SimplicalComplex
from .distance import distance_matrix, points_max_distance

# Vietoris-Rips filtering

# The Vietoris-Rips complex can be calculated from distances alone,
# which both simplifies calcualations, and make it possible to apply
# the filtering on other distances than euclidan.
#

class RipsComplex:


    def __init__(
        
        self, max_dim=2, max_radius=None, max_simplices=None, num_steps=500, verbose=1, input_distance_matrix =True
    ):
        self.verbose = verbose
        # Limit number of simplices by different means
        self.max_dim = max_dim
        self.max_radius = max_radius
        self.max_simplices = max_simplices
        self.num_steps = num_steps
        # We assume input to rips is a distance matrix
        self.input_distance_matrix = input_distance_matrix

        self.debug_test = True

    # Fit from distance matrix
    # (Will not work with NaNs, prefiltered values should be set to inf)
    def fit(self, X_dist):

        if self.input_distance_matrix == False:
            X_dist = distance_matrix(X_dist) 

        X_dist_lower = np.tril(X_dist)
        if self.max_radius is None:
            max_radius = np.nanmax(X_dist)
        else:
            max_radius = self.max_radius
        # Linear breaks for now. Try area/volumebased for finer resolution?
        # Or unique, sorted distances?
        breaks = np.linspace(0, max_radius, num=self.num_steps, endpoint=True)
        # breaks = np.sort(np.unique(X_dist_lower))

        # Simplex added counter (index to boundary matrix)
        sidx = 0
        # Keep an index of all the added simplices, index and filtration value
        simplex_collection = {}
        # Add the points as 0-simplices
        for i in range(0, len(X_dist)):
            # Value of a simlex is (index, dimension, filter distance)
            simplex = frozenset([i])
            simplex_collection[simplex] = (i, 0, 0)
            sidx += 1

        eps_prev = 0
        t_start = time()
        for t, eps in enumerate(breaks):
            # Find all new edges
            within = np.where((X_dist_lower > eps_prev) & (X_dist_lower <= eps))
            # If no new points are within range, skip to next filtration value
            if len(within[0]) == 0:
                continue
            
            # Edges in current filtration value
            edges = [frozenset({i, j}) for i, j in np.transpose(within)]
            edge_lengths = [points_max_distance(X_dist, edge) for edge in edges]
            edge_tuple = [
                (edge, edist) for edist, edge in sorted(zip(edge_lengths, edges))
            ]
            edges = [ edge for edge,edist in edge_tuple ]

            for edge, edge_length in edge_tuple:
                simplex_collection[edge] = (
                    sidx,
                    1,
                    edge_length,
                )
                sidx += 1
            # Find the higher order simplices
            simplices_added_prev_dim = edges  # Simplices added lower dimension
            simplices_new = []  # Simplices added current dimension
            for dim in range(2, self.max_dim + 1):
                for simplex in simplices_added_prev_dim:
                    # For current distance, check if any new nodes have reached
                    # epsilon-distance, and add these to d+1 dimensional simplices
                    point_dist = np.nanmax(X_dist[:, tuple(simplex)], axis=1)
                    within = np.where((point_dist > 0) &  (point_dist <= eps))
                    
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
                            # discrete values from the Rips iterations
                            simplex_max_dist = points_max_distance(X_dist, new_simplex)
                            simplex_collection[new_simplex] = (
                                sidx,
                                dim,
                                simplex_max_dist,
                            )
                            sidx += 1
                simplices_added_prev_dim = simplices_new
            eps_prev = eps
            if self.verbose > 1:
                print(f"eps={eps}, n={len(simplex_collection)}")
            if (
                self.max_simplices is not None
                and len(simplex_collection) > self.max_simplices
            ):
                if self.verbose > 0:
                    print(f"Reached max number of simplices ({self.max_simplices}) at eps={eps}")
                break
        if self.verbose > 0:
            print(f"Rips filtration: {len(simplex_collection)} simplices, {round(time() - t_start, 2)} sec.")
        
        self.simplical_complex = SimplicalComplex(simplex_collection)
        return self.simplical_complex

    def transform(self):
        # Only transform self here.
        return self.simplical_complex.birth_death_pairs()

    def fit_and_transform(self, X):
        self.fit(X)
        return self.transform()
        



