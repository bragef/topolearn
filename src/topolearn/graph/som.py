import numpy as np
import networkx as nx
from random import sample


class SelfOrganisingMaps:
    def __init__(self, graph,  k=1000, alpha=0.05, batch_size=0.2, conv_rate = 0.001, debug=1):
        self.k = k
        self.alpha = alpha
        self.graph = graph
        self.batch_size = batch_size
        self.debug = debug
        self.conv_rate = conv_rate # Stop when Δer/err < conv_rate
        if not isinstance(graph, nx.Graph):
            raise TypeError("Input graph must be a networkx.Graph. Example: graph=nx.triangular_lattice_graph(10,10)")

    def fit(self, X):
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
            if self.debug > 0 and epoch % 100 == 0:
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

    def transform():
        pass
    
