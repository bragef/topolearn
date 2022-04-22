import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from itertools import combinations
from .simpcomplex import SimplicalComplex
from .distance import distance_matrix, points_max_distance


# AlphaComplex filtering
class AlphaComplex:
    """Alpha Complex

    Examples
    --------
    >>> import numpy as np
    >>> from topolearn.util import plot_graph_with_data, plot_persistance_diagram
    >>> from topolearn.simpcomplex import AlphaComplex
    >>> from sklearn.datasets import make_moons, make_circles

    >>> X1,_ = make_circles(noise=0.125,  n_samples=400, random_state=50)
    >>> X2,_ = make_circles(noise=0.125,  n_samples=200, random_state=50)
    >>> X = np.vstack([X1, X2*0.5 + [2,0]])

    >>> learner = AlphaComplex()
    >>> simplices = learner.fit(X)
    >>> homologies = learner.transform()

    >>> plot_graph_with_data(simplices.graph(X), X, axis=True)
    >>> plot_persistance_diagram(homologies)
    """    

    def __init__(self, verbose=1, max_radius = None):
        self.verbose = verbose
        self.max_radius = max_radius

    def fit(self, X, X_dist = None):
        """Fit an alpha complex

        Parameters
        ----------
        X : matrix
            Feature matrix
        X_dist : matrix, optional
            Feature distance matrix. If not supplied, Euclidian distance will be used
            as edge weights

        Returns
        -------
        SimplicalComplex
            Fitted simplical complex

        """     

        DG = Delaunay(X)

        # Distance matrix between points used for ball radius. We use euclidian 
        # distance here, for a weighted alpha complex, this can be replace with 
        # by weighted values.
        if X_dist is None:
            X_dist = distance_matrix(X)

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

        # Build a simplical complex similar to what we do in ripssimplex.py
        simplex_collection = {}
        # Sort the simplices by (radius,dimension) to get the filtration order the simplices
        # are added to simplical complex 
        simplices_sorted = sorted(simplex_maxdist.items(), key=lambda x: (x[1], len(x[0])))

        for sidx, (simplex, eps) in enumerate(simplices_sorted):
            if self.max_radius is not None and eps > self.max_radius: 
                continue
            simplex_collection[simplex] = (sidx, len(simplex) - 1 , eps)
        
        self.simplex_maxdist = simplex_maxdist
        self.simplical_complex = SimplicalComplex(simplex_collection)

        return self.simplical_complex

    def transform(self):
        """Return the persistance pairs from the fitted complex

        Returns
        -------
        list of birth death pairs
        """        
        # Only transform self here; the fit_and_transform_method make more sense.
        return self.simplical_complex.birth_death_pairs()

    def fit_and_transform(self, X):
        """Fit an alpha complex and return the birth-death pairs

        Parameters
        ----------
        X : Feature matrix

        Returns
        -------
        list of birth death pairs
        """        
        self.fit(X)
        return self.transform()
        
