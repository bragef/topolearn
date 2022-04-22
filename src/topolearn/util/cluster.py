from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from numpy.random import random_sample
import numpy as np
import matplotlib.pyplot as pl


class KMeansGap:
    """Fit a KMeans model with k-size selection using Tibshirani's Gap statstic 

    Parameters
    ----------
    max_k : int, optional
        Maximum number of clusters, by default 10
    gap_iter : int, optional
        _description_, by default 10

    Attributes
    ----------
    model: sklearn.cluster.KMeans
        Selected model
    n_clusters:
        Number of clusters

    Notes
    -----
    (Why is this not available in sklearn?)
    """        

    def __init__(self, max_k = 10, gap_iter = 10):
        self.max_k = max_k
        self.gap_iter =gap_iter 

    def fit(self, X):

        max_k = min(X.shape[0]-1, self.max_k)
        if max_k < 1:
            return None
            
        k_range = range(1,max_k + 1)
        b_max = self.gap_iter 

        # Boundary box for data set 
        xmin = np.min(X, axis=0)
        xmax = np.max(X, axis=0)

        w_k = np.zeros(len(k_range))   # Log-inertia current clusters
        w_b = np.zeros(len(k_range))   # Sum log inertia random clusters
        gap = np.zeros(len(k_range))   # gap values

        # Save all base models until finished 
        models = list() 

        # Todo: we could probably reuse X_rand between the k's
        for (i, k) in enumerate(k_range):
            base_clusterer = MiniBatchKMeans(n_clusters=k)
            base_clusterer.fit(X)
            models.append(base_clusterer)
            if base_clusterer.inertia_ > 0:
                w_k[i] = np.log( base_clusterer.inertia_)
                
                for j in range(b_max):
                    X_rand = (xmax - xmin) * random_sample(size=X.shape) + xmin 
                    w_b[i] += np.log(MiniBatchKMeans(n_clusters=k).fit(X_rand).inertia_ )

            gap[i] = w_b[i]/b_max - w_k[i]

        best_model = np.argmax(gap)
        
        self.w_b = w_b
        self.w_k = w_k
        self.gap = gap
        self.model = models[best_model]
        self.n_clusters = best_model + 1

        return self.model



# Cluster algorithms.

# Hierarchical clustering with agglomerative clustering generally works well,
# but need some tuning of the threshold parameter to make sense.
# 
def cluster_agglomerative(X, distance_threshold=0.1):
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        compute_full_tree=True,
        linkage="single",
    )
    cluster_labels = model.fit_predict(X)

    return cluster_labels


# Gaussian mixture model.
# Did not work well at all. The current version only supports agglomerative clustering.
def cluster_gaussian(X, max_clusters=10):
    bic = []
    best_gmm = None
    lowest_bic = np.infty
    covariance_type = ["spherical", "tied", "diag", "full"]
    for cov in covariance_type:
        for n_components in range(1, max_clusters):
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(
                n_components=n_components, covariance_type=cov
            )
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    best_fit = best_gmm.fit_predict(X)

    return best_fit

