import numpy as np


# The Vietoris-Rips complex can be calculated from distances alone,
# which both simplifies calcualations, and make it possible to apply
# the filtering on other distances than euclidan.
# Create a distance matrix from input X feature matrix.
def calc_distance_matrix(X):
    dist_matrix = np.zeros((X.shape[0], X.shape[0]))
    for j, xj in enumerate(X):
        dist_matrix[j, :] = np.linalg.norm(X - X[j, :], axis=1)
    return dist_matrix


# 
def points_max_distance(X_dist, simplex):
    return np.max(X_dist[np.ix_(tuple(simplex), tuple(simplex))])

def points_max_distance_edge(X_dist, simplex):
    # Awkward numpy syntax location of max index.
    t = tuple(simplex)      #  Points of simplex = index in distance matrix
    x_idx = np.ix_(t, t)    #  Simplex to index in X-matrix 
    # (If more than one match, np.argmax only returns first.)
    max_idx = np.unravel_index(np.argmax(X_dist[x_idx], axis=None), (len(t),len(t)))
    # .. and finally convert from t-index to x_dist-index and return the 
    #  edge-point-tuple
    return (t[max_idx[0]], t[max_idx[1]])

    


