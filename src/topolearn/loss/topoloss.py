from ..simpcomplex import distance_matrix, points_max_distance_edge
import numpy as np


class TopologicalLoss:
    """Loss function for topological autoencoder

    Parameters
    ----------
    filtration: RipsComplex
        

    Example
    -------
    The intention is that TopologicalLoss should be used to calculate the topological loss
    inside a training loop. See the autoencoder.ipynb notebook for a full example.

    >>> from topolearn.simpcomplex import RipsComplex
    >>> from topolearn.loss import TopologicalLoss
    >>> topoloss = TopologicalLoss(
    >>>     filtration=RipsComplex(max_dim=1, max_radius=0.8, verbose=0),
    >>> )
    >>> # Calculate loss 
    >>> # loss = topoloss(X, Z)

    Notes
    -----
    Implements loss calculations for a topological autoencoder, as described in

    Moor, Michael, Max Horn, Bastian Rieck, and Karsten Borgwardt. 2020. 'Topological Autoencoders'. 
    In Proceedings of the 37th International Conference on Machine Learning, 7045-54. PMLR.
    """
    def __init__(self, filtration, max_dim):
        self.filtration = filtration
        self.max_dim = max_dim

    # Return value is a tuple of two arrays with x and y coordinates
    # of the A matrix, suitable for numpy indexing.
    def find_critical_edges(self, A, max_dim=1):
        """Find the index of the edges responsible for birth or death of a homology.

        Parameters
        ----------
        A : matrix
            Distance matrix
        max_dim : int, optional
            Maximal dimension homologies, default 1
            (Note that this should be set in the SimplicalComplex)

        Returns
        -------
        indices: (shape=(n_edges,1), shape=(n_edges,1))
            Indices of edges in distance matrix
        """        
        # Homology calculations
        simplices = self.filtration.fit(A)
        bdpairs = simplices.birth_death_pairs(verbose=0)

        # Save the edges in numpy-index-friendly coordinates
        edges_x = []
        edges_y = []

        for b, d, dim, b_value, d_value in bdpairs:
            # Diagonal values are just noise
            if b_value == d_value or dim > max_dim:
                continue
            # Birth-edges
            if dim > 0:
                s_dim, s_birth, simplex = simplices.get_simplex(b)
                edge = points_max_distance_edge(A, simplex)
                edges_x.append(edge[0])
                edges_y.append(edge[1])
            # Same for death-edges
            if d is not None:
                s_dim, s_birth, simplex = simplices.get_simplex(d)
                edge = points_max_distance_edge(A, simplex)
                edges_x.append(edge[0])
                edges_y.append(edge[1])

        return (edges_x, edges_y)

    def calculate_topo_loss(self, X, Z):
        """Calculate the topolgical loss function from two feature matrices 
   
        Parameters
        ----------
        X : matrix (shape=(n_samples, dim_inputl))
            Feature matrix original dimension
        Z : matrix (shape=(n_samples, dim_encoded))
            Feature matrix in encoded layer 

        Returns
        -------
        float
            Topological loss 
        """        
        assert len(X) == len(Z)

        Ax = distance_matrix(X)
        Az = distance_matrix(Z)

        edges_z = self.find_critical_edges(Az)
        edges_x = self.find_critical_edges(Ax)

        Lxz = np.linalg.norm(Ax[edges_x] - Az[edges_x]) ** 2
        Lzx = np.linalg.norm(Az[edges_z] - Ax[edges_z]) ** 2
        L = 1 / 2 * (Lxz + Lzx)

        # rho = Ax[edges_x] - Az[edges_x]
        return L

    def __call__(self, input, output):
        return self.calculate_topo_loss(input, output)
