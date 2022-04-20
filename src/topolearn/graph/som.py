import numpy as np
import networkx as nx
from random import sample


class SelfOrganisingMaps:
    def __init__(self, graph,  k=1000, alpha=0.05, batch_size=0.2, conv_rate = 0.001, verbose=1):
        """Self Organising Map graph learner

        Parameters
        ----------
        graph : networkx.graph
            Graph to fit. (See examples)
        k : int, optional
            Number of epochs
        alpha : float, optional
            Learning rate, by default 0.05
        batch_size : float, optional
            Batch size. Size of random sample used for each epoch.
        conv_rate : float, optional
            Convergence rate, stop at Δer/err < conv_rate
        verbose: int

        Attributes
        ----------
        graph: 
            Fittet graph

        Examples
        --------
        The networkx graph generators are useful to to specify an initial graph for fitting. 
    
        >>> import networkx as nx
        >>> from topolearn.graph import som
        >>> from topolearn.util import plot_graph_with_data
        >>> from sklearn.datasets import make_circles
        >>> X, y = make_circles(noise=0.05, n_samples=500)

        >>> # Example 1: Fit a 5 x 5 hexagonal grid
        >>> geometry = nx.hexagonal_lattice_graph(5, 5) 
        >>> learner = som.SelfOrganisingMaps(graph=geometry, alpha=0.001)
        >>> graph = learner.fit(X)
        >>> plot_graph_with_data(graph, X)
        
        >>> # Example 2: Fit a circle to the same data set
        >>> geometry = nx.cycle_graph(20)
        >>> learner = som.SelfOrganisingMaps(graph=geometry, alpha=0.001)
        >>> graph = learner.fit(X)
        >>> plot_graph_with_data(graph, X)

        Notes
        -----
        Algorithm described in 
        Kohonen, Teuvo. 1982. ‘Self-Organized Formation of Topologically Correct Feature Maps’. 
        Biological Cybernetics 43 (1): 59–69.

        """        
        self.k = k
        self.alpha = alpha
        self.graph = graph
        self.batch_size = batch_size
        self.verbose = verbose
        self.conv_rate = conv_rate # Stop when Δer/err < conv_rate
        if not isinstance(graph, nx.Graph):
            raise TypeError("Input graph must be a networkx.Graph. Example: graph=nx.triangular_lattice_graph(10,10)")

    def fit(self, X):
        """Fit at Generative Gaussian Graph model

        Parameters
        ----------
        X : (array, shape = [n_samples, n_features])

        Returns
        -------
        networkx.graph
            Fitted graph. Weights saved as ``w`` attribute of nodes.
        """
        # We want neat integer nodeids we can use as matrix indices
        graph = nx.convert_node_labels_to_integers(self.graph)
        # Initialise the node positions.
        center = np.mean(X, axis=0)
        scale = np.std(X)
        # Eigenvector based node placement
        layout = nx.spectral_layout(graph, center=center, scale=scale, dim=X.shape[1])
        n_nodes = len(graph.nodes)
        # Allocate arrays
        distance_matrix = np.zeros((X.shape[0], n_nodes))
        weights = np.zeros((n_nodes, X.shape[1]))

        # Create a N (axis=0) x k (axis=1) matrix of distances
        for (i, (n1, point)) in enumerate(layout.items()):
            assert i == n1, "Bug: unstable indexing (python < 3.6?)"
            distance_matrix[:, n1] = np.linalg.norm(X - point, axis=1)
            weights[i, :] = point

        # Time to learn!
        error = prev_error = 0
        batch_count = min(round(X.shape[0] * self.batch_size), X.shape[0])
        for epoch in range(0, self.k):
            # Sample input vectors
            input_batch = sample(range(0, X.shape[0]), batch_count)
            for i in input_batch:
                point = X[i, :]
                node_closest = np.argmin(distance_matrix[i, :])
                # Todo: Replace with a configurable neighborhood-function
                # We here just use a neighborhood function of 1 for neighbor nodes, 0 else. 
                neighbors = list(graph.neighbors(node_closest))
                for n1  in [node_closest] + neighbors:
                    weights[n1, :] = weights[n1, :] + self.alpha * 1 * (
                        point - weights[n1, :]
                    )
                    # Update distances for affected node
                    distance_matrix[:, n1] = np.linalg.norm(X - weights[n1, :], axis=1)
            prev_error = error
            # Reconstruction error is just the sum of the shortest distance for each point
            error = np.sum(np.min(distance_matrix, axis=1)**2)
            if self.verbose > 0 and epoch % 100 == 0:
                print(f"Epoch: {epoch}, error: {error}")
            if error == 0 or abs(prev_error - error) / error < self.conv_rate:
                print(f"Convergence in {epoch} steps")
                break

        # Add weights to learned graph
        for i in range(0, len(weights)):
            graph.nodes[i]["w"] = weights[i, :]

        self.graph = graph
        self.distance_matrix = distance_matrix

        self.error = error

        return self.graph

    def transform(self, X, return_nodeids=False):
        """Transform data to nearest weights in fitted model

        Parameters
        ----------
        X : (array, shape = [n_samples, n_features])
            Input features
        return_nodeid: bool
            If return_nodeid is set to True, return the id of the node in the graph object,
            otherwise return the weights.
        Returns
        -------
        y: (array, shape=[n_samples, n_features])
            Coordinates of closest point on 
        """
        weights = [d["w"] for _,d in self.graph.nodes(data=True)]
        idx = [ np.argmin( np.linalg.norm(p - weights, axis=1)) for p in X ]
        weights_out  = [ weights[i] for i in idx  ]

        if return_nodeids: 
            return idx
        else:
            return weights_out


