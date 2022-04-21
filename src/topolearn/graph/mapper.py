import numpy as np
import warnings
import networkx as nx

from ..util.cluster import cluster_agglomerative
from ..util.cluster import KMeansGap

def filter_d1(X):
    """Filter dim 1
    """
    # Default filter function is first dimension of feature matrix
    return X[:,0]

## 
## 
##
## 


class Mapper:
    """Use the Mapper algorithm to learn a Reeb graph

    Parameters
    ----------
    n_intervals : int, optional
         by default 10
    filter_fun : function(X) -> f , optional
        Filter function. Default is first column of data matrix  
    min_clustersize : int, optional
        If set, remove clusters below this size.

    Attributes
    ----------
    graph: networkx.graph
        Learned graph. Node labels are tuples (int, int) with filtration 
        index and cluster index within filtration interval.
    bin_ranges: (array, shape=(n_intervals,2))
        Filtration intervals

    Examples
    --------
    >>> X1, _ = make_circles(noise=0.1, random_state=1, n_samples=1000)
    >>> learner = Mapper(n_intervals = 20)
    >>> graph = learner.fit(X)
    >>> plot_graph_with_data(graph, X, alpha=1)    
    >>> # nodes = learner.transform(X)

    Notes
    -----
    The current version uses KMeans as clustering, and select the number of clusters 
    using the Gap-statistic. I originally wrote the code using Gaussian Mixtures or
    Agglomerative Clustering. These are available in ``topolearn.util.cluster``, 
    but will need som retrofitting to be functional again. 

    The mapper algorithm is described in
    Singh, Gurjeet, Facundo Memoli, and Gunnar Carlsson. 2007. Topological Methods 
    for the Analysis of High Dimensional Data Sets and 3D Object Recognition. 
    The Eurographics Association.
    """   

    def __init__(
        self, n_intervals=10, verbose=0, filter_fun=filter_d1, min_clustersize = 1
    ):

        self.n_intervals = n_intervals
        self.verbose = verbose
        # self.cluster_mindistance = cluster_mindistance
        self.filter_fun = filter_fun
        self.overlap = 0.33
        self.min_clustersize = min_clustersize

    def fit(self, X):
        """Fit a Mapper graph

        Parameters
        ----------
        X : (array, shape = [n_samples, n_features])
            Data matrix

        Returns
        -------
        networkx.graph
            Learned graph
        """        
        # Find the bins and retrieve the indices in the X matrix for each bin
        bin_ranges, bin_indices  = self._split_intervals(X,self.n_intervals, self.overlap)

        self.bin_ranges =  bin_ranges
        self.bin_indices = bin_indices 

        clusters = self._find_clusters(X)
        # Connect the clusters using single linkage
        self.graph = self._connect_clusters()
        
        return self.graph


    # Split into overlapping covers.
    # Returns a list of indices for each set.
    def _split_intervals(self, X, n_intervals=10, overlap=0.25):

        # Minimal sanity check.
        if abs(overlap) > 1:
            warnings.warn("overlap should be between 0 and 1")

        # Get the range of filter values.
        filter_values = np.array(self.filter_fun(X))
        fmin = filter_values.min()
        fmax = filter_values.max()

        interval_length = (fmax - fmin) / (n_intervals * (1 - overlap) + overlap)
        overlap_length = overlap * interval_length

        # Return a list of features in each range
        bin_features  = list()
        bin_ranges = list()
        bin_start = fmin
        for i in range(0, n_intervals):
            bin_start = fmin + i * (interval_length - overlap_length)
            bin_end = bin_start + interval_length
            interval_idx = (filter_values > bin_start) & (filter_values < bin_end)
            bin_ranges.append((bin_start,bin_end))
            bin_features.append(interval_idx)
        return (bin_ranges, bin_features)

    # Apply selected clustering algorithm to the covers.
    # 
    def _find_clusters(self, X):
        cluster_id = 0
    

        self.clusters = list()
        self.clusters_centers = {}
        self.cluster_models = list()

        cluster_method = KMeansGap(gap_iter = 5)
        for interval_i, interval_idx in enumerate(self.bin_indices):
            clusters_local = list()
            cover = X[interval_idx]
            # cluster_labels = cluster_gaussian(cover,  max_clusters = 5)
            # cluster_labels = cluster_agglomerative(cover, distance_threshold=self.cluster_mindistance)
            if cover.shape[0] == 0:
                continue
            cluster_model  = cluster_method.fit(cover)
            cluster_labels = cluster_model.predict(cover)
            # Features as index of X matrix
            points = np.where(interval_idx)[0]
            # Point set for each
            for label in np.unique(cluster_labels):
                # Add identier valid across intervals
                cluster_id = (interval_i, label)
                # Cluster is tuple(id, pointset)
                cluster_pointset = set(points[cluster_labels == label])
                if len(cluster_pointset) >= self.min_clustersize:
                    clusters_local.append((cluster_id, cluster_pointset))
                    self.clusters_centers[cluster_id] = cluster_model.cluster_centers_[label]
            self.clusters.append(clusters_local)
            self.cluster_models.append(cluster_model)
        return self.clusters
        

    # Given the list of clusters find the edges.
    # Use single linkage here, other linkage methods may make more sense
    # depending on the data.
    # Returns nx.Graph object
    def _connect_clusters(self):
        prev_interval = []
        graph = nx.Graph()
        for interval in self.clusters:
            for cluster in interval:
                if self.clusters_centers is not None:
                    # Add plotting coordinates: x = filtration value, y = KMeans cluster center coordinates
                    w =  np.array([self.bin_ranges[cluster[0][0]][0], self.clusters_centers[cluster[0]][1]])
                else: 
                    w =  np.array([np.nan,np.nan])
                graph.add_node(cluster[0], w=w)
                if self.verbose > 0:
                    print(f"  cluster {cluster[0]} ({len(cluster[1])})")
                for prev_cluster in prev_interval:
                    # Single linkage: Exists point in both clusters
                    # Should probably try a more noise-resistant linkage 
                    overlap = len(cluster[1] & prev_cluster[1])
                    if overlap > 0:
                        graph.add_edge(cluster[0], prev_cluster[0])
                        if self.verbose > 0:
                            print(f"    ({cluster[0]}->{prev_cluster[0]}) ({overlap}) ")
            prev_interval = interval

        return graph

    def transform(self, X):
        """Transform data to closest mapper graph node. 

        Parameters
        ----------
        X : (array, shape = [n_samples, n_features])
            Data matrix

        Returns
        -------
        (int, int)
            Node indices to the learned graph. 

        """        
        # Filtration values
        filter_values = np.array(self.filter_fun(X))
        interval_start = np.array(self.bin_ranges)[:,0]
        filter_idx = [ np.where(interval_start <= f )[0][-1] for f in filter_values ]
        labels = list()

        for idx, xd in zip(filter_idx, X):
            # Get predicted label. Award numpy indexing 
            # to account for 1D data.
            label_local = self.cluster_models[idx].predict([xd])[0] 
            # Add idx to get global label
            labels.append((idx, label_local))
        
        return labels





        





