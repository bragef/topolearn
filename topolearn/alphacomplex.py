import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from itertools import combinations
from .simpcomplex import SimplicalComplex
from .rips import calc_distance_matrix

# AlphaComplex filtering
class AlphaComplex:

    def __init__(self, debug=1):
        self.debug = debug

    def fit(self, X, max_radius=None, max_simplices=None, num_steps=10):
        DG = Delaunay(X)

        # Distance matrix between points used for ball radius. We use euclidian 
        # distance here, for a weighted alpha complex, this should be replaced
        # by weighted values.
        X_dist = calc_distance_matrix(X)

        # Number of points in delaney calculated simplices (dim+1)
        rdim = DG.simplices.shape[1]  #

        # Max distance between vertices in simplex.
        # Initialise with 0-simplices, which need no computatations
        simplex_maxdist = {
            frozenset({nodeid}): 0.0 for nodeid, w in enumerate(DG.points)
        }

        # Iterate over the simplices and sub-simplices and calculate at which
        # distance they appear
        for simplex in DG.simplices:
            simplex_set = frozenset(simplex)
            if simplex_set in simplex_maxdist:  # Already added
                continue
            simplex_maxdist[simplex_set] = points_max_distance(X_dist, simplex_set)
            for r in range(2, rdim):  # r=2 for two dimensional data
                for subsimplex in combinations(simplex_set, r):
                    subsimplex_set = frozenset(subsimplex)
                    if subsimplex_set not in simplex_maxdist:
                        simplex_maxdist[subsimplex_set] = points_max_distance(
                            X_dist, subsimplex_set
                        )

        # Build a simplical complex similar to what we do in rips.py
        simplex_collection = {}
        # Sort the simplices by (radius,dimension) to get the order the simplices
        # are added to simplical complex 
        simplices_sorted = sorted(simplex_maxdist.items(), key=lambda x: (x[1], len(x[0])))

        for sidx, (simplex, eps) in enumerate(simplices_sorted):
            if max_radius is not None and eps > max_radius: 
                continue
            simplex_collection[simplex] = (sidx, len(simplex) - 1 , eps, eps)
        
        self.simplex_maxdist = simplex_maxdist
        self.simplical_complex = SimplicalComplex(simplex_collection)

        return self.simplical_complex


def points_max_distance(X_dist, simplex):
    return np.max(X_dist[np.ix_(tuple(simplex), tuple(simplex))])
