import numpy as np
import pandas as pd
import warnings 

import networkx as nx

from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering


class Mapper():

    def __init__(
        self,
        n_intervals = 10,
        min_clustersize = 2,
        debug = 0
            
    ):  
        self.min_clustersize = min_clustersize
        self.n_intervals = n_intervals  
        self.debug = debug


    # Filter function is first dimension of feature matrix unless
    # specified otherwise.
    def filter_x(self, X):
        return X[:,0]

    # Split into overlapping covers. 
    # Returns a list of indices for each set.
    def split_intervals(self, X, n_intervals = 10, overlap = 0.25):

        # Minimal sanity check. 
        if abs(overlap) > 1:
            warnings.warn("overlap should be between 0 and 1")

        # Get the range of filter values. 
        filter_values = np.array(self.filter_x(X))
        fmin = filter_values.min()
        fmax = filter_values.max()

        interval_length = (fmax - fmin) / (n_intervals * (1-overlap) + overlap) 
        overlap_length = overlap * interval_length

        # Return a list of features in each range
        bins = list()
        bin_start = fmin
        for i in range(0, n_intervals): 
            bin_start = fmin + i * (interval_length - overlap_length)
            bin_end = bin_start + interval_length
            interval_idx =  ( filter_values > bin_start ) & (filter_values < bin_end) 
            bins.append(interval_idx)
        return bins

    # Apply selected clustering algorithm til the covers.
    # Return list(    )
    def find_clusters(self, X, interval_indices):
        cluster_id = 0
        clusters = list()
        for interval_idx in interval_indices:
            cl = list()
            cover = X[interval_idx]
            #cluster_labels = cluster_gaussian(cover,  max_clusters = 5)
            cluster_labels  = cluster_agglomerative(cover)
            # Features as index of X matrix 
            points = np.where(interval_idx)[0] 
            # Point set for each 
            for label in np.unique(cluster_labels):
                # Add identier valid across intervals
                cluster_id += 1
                # Cluster is tuple(id, pointset)
                cluster_pointset = set(points[cluster_labels == label])
                if len(cluster_pointset) >= self.min_clustersize: 
                    cl.append((cluster_id, cluster_pointset))
            clusters.append(cl)
 
        return clusters

    # Given the list of clusters found by the algorithm, find the edges. Return 
    # nx.Graph object 
    def connect_clusters(self, clusters):
        prev_interval = []
        graph = nx.Graph()
        for interval in clusters:
            for cluster in interval:
                graph.add_node(cluster[0])
                # graph.set_node_attributes(cluster[0],  {"size":len(cluster[1])})
                if self.debug > 0: 
                    print(f"  cluster {cluster[0]} ({len(cluster[1])})")
                for prev_cluster in prev_interval:
                    overlap = len( cluster[1] & prev_cluster[1])
                    if overlap > 0:
                        graph.add_edge(cluster[0], prev_cluster[0])
                        if self.debug > 0 :
                            print(f"    ({cluster[0]}->{prev_cluster[0]}) ({overlap}) ")
            prev_interval = interval

        return graph

 

# Cluster algorithms.
# Idea: predict with previous cloud for edges. 
def cluster_gaussian(X,  max_clusters = 10):
    bic=[]
    best_gmm = None
    lowest_bic = np.infty
    covariance_type = ['spherical', 'tied', 'diag', 'full']
    for cov in covariance_type:
        for n_components in range(1, max_clusters):
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(
                n_components=n_components, 
                covariance_type=cov
            )
            gmm.fit(X)

            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    best_fit = best_gmm.fit_predict(X)

    return best_fit


def cluster_agglomerative(X, distance_threshold = 0.1):
    model = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=0.1,
        compute_full_tree=True,
        linkage="single")
    cluster_labels = model.fit_predict(X)

    return cluster_labels
