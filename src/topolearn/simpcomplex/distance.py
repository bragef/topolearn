import numpy as np
from scipy.spatial import KDTree


def distance_matrix(X):
    """Calculate euclidian distance matrix
    
    Parameters
    ----------
    X : matrix
        Feature matrix

    Returns
    -------
    numpy matrix
        Full (symmetric) distance matrix 
    """    
    dist_matrix = np.zeros((X.shape[0], X.shape[0]))
    for j, xj in enumerate(X):
        dist_matrix[j, :] = np.linalg.norm(X - X[j, :], axis=1)
    return dist_matrix

def distance_matrix_mreach(X, k):
    """Calculate the mutual reachability distance matrix of order k   
    
    Parameters
    ----------
    X : matrix
        Feature matrix
        
    k : int
        Core distance order. 

    Returns
    -------
    numpy matrix
        Distance matrix

    Notes
    -----
    The mutual reachability distance is a density weighted distance measure, defined as
    $d_{\mathrm{mreach-}k}(a,b) = \max \{\mathrm{core}_k(a), \mathrm{core}_k(b), d(a,b) \}$, where
    $d(a,b)$ is the euclidian distance, and $\mathrm{core}_k(b)$ is the distance to the k't nearest
    neighbour. See  https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html.
    """
    
    # d_mreach-k(a,b) = max(core_k(a), core_k(b), d(a,b))
    # d(a,b)-matrix
    X_dist = distance_matrix(X)
    # Distance to the k'th nearest point, i.e. core_k()
    tree = KDTree(X)     
    k_dist, _ = tree.query(X, k=[k])  
    # Create a matrix of {max(core_k(a), core_k(b)}
    dim = X.shape[0]
    core_mat = np.maximum.outer(k_dist, k_dist).reshape((dim,dim))
    # And finally d_mreach as max of these
    X_mreach_k = np.maximum(X_dist, core_mat)
    # Distance to self is zero
    np.fill_diagonal(X_mreach_k, 0)

    return X_mreach_k    

def points_max_distance(X_dist, simplex):
    """Maximum pointwise distance in simplex

    Parameters
    ----------
    X_dist : matrix 
        Distance matrix 
    simplex : set 
        A set of indices of the points in the simplex

    Returns
    -------
    float
        Maximal distance between any two points in the simplex
    """  
    
    return np.nanmax(X_dist[np.ix_(tuple(simplex), tuple(simplex))])

def points_max_distance_edge(X_dist, simplex):
    """Index of furthest points in simplex

    arg max points_max_distance(X_dist, simplex)

    Parameters
    ----------
    X_dist : matrix
        Distance matrix
    simplex : set 
        A set of indices of the points in the simplex
    Returns
    -------
    (int, int)
        A tuple with the index of the furthest point in the simplex
    """    
    # Awkward numpy syntax location of max index.
    t = tuple(simplex)      #  Points of simplex = index in distance matrix
    x_idx = np.ix_(t, t)    #  Simplex to index in X-matrix 
    # (If more than one match, np.argmax only returns first.)
    max_idx = np.unravel_index(np.argmax(X_dist[x_idx], axis=None), (len(t),len(t)))
    # .. and finally convert from t-index to x_dist-index and return the 
    #  edge-point-tuple
    return (t[max_idx[0]], t[max_idx[1]])

    


