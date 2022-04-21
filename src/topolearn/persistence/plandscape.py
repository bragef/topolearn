
import numpy as np
import matplotlib.pyplot as pl

class PersistenceLandscape:
    """Persistence Landscapes

    Parameters
    ----------
    resolution : int, optional
        Default 100
    max_m : float, optional
        Maximum value for the landscape function. Used to find the grid resolution, 
        if the values are used for statistics or comparison, these should be set to
        a shared value for all the landscapes.

    Attributes
    ----------
    matrix (array, shape=(resoution, n_persist))
        Matrix with persistence landscape values. Note that only the horisontal resolution
        is fixed.

    Examples
    --------
    >>> from topolearn.simpcomplex import AlphaComplex
    >>> from topolearn.persistence import PersistenceLandscape
    >>> from sklearn.datasets import make_circles

    >>> X, _ = make_circles(noise=0.2,  n_samples=1000, random_state=50)
    >>> learner = AlphaComplex()

    >>> simplices = learner.fit(X)
    >>> pairs = simplices.birth_death_pairs()
    >>> p = PersistenceLandscape(resolution=400)
    >>> p.fit(pairs,dim=1)
    >>> p.plot()
    >>> print(p.matrix)

    Notes
    -----
    Perstence landscapes are described in 
    Bubenik, Peter. 2015. ‘Statistical Topological Data Analysis Using 
    Persistence Landscapes’. The Journal of Machine Learning Research 16
    """    


    def __init__(self, resolution = 100, max_m = None):
        self.resolution = resolution
        self.max_m = max_m
        

    # For statistics to make sense, max_m need to be set beforehand.
    def fit(self, pairs, dim = 1, resolution = 100, max_m = None):
        """Find the persistence landscape for a homology dimension

        Parameters
        ----------
        pairs : array
            List of birth-death pairs as returned by SimplicalComplex.birth_death_pairs)
        dim : int, optional
            _description_, by default 1
            
        Returns
        -------
        (array, shape=(resoution, n_persist))
            Matrix with persistence landscape values
   
        """
        pairs = np.array(pairs)
        sdim = np.array(pairs[:, 2], dtype=int) 
        b = np.array(pairs[:, 3], dtype=float)  # Births
        d = np.array(pairs[:, 4], dtype=float)  # Deaths)
        
        # We only need the pairs with non-zero and finite persistance
        points = np.where(np.isfinite(d) & ((d-b) > 0) & (sdim == dim))

        m = (d[points] + b[points])/2   # Mid-life
        h = (d[points] - b[points])/2   # Half-life         
        npoints = m.shape[0]

        # Calculate a reasonable uppper limit guaranteed to include
        # the full triangular function. This should be set beforehand
        # to a common value if the landscapes are used for statistics!
        max_m = self.max_m
        if max_m is None: 
            max_m = np.max(m) + np.max(h)
        grid_m = np.linspace(0, max_m, self.resolution)
        lambda_mat = np.zeros((npoints, self.resolution))

        # Triangular function for each point 
        for i, (u_m, u_h) in enumerate(zip(m, h)):
            lambda_mat[i] = np.maximum( u_h - np.abs(u_m - grid_m), 0)
        # Sort triangles to get λ(m,h)
        landscape = -np.sort(-lambda_mat, axis=0)  

        self.grid_m = grid_m
        self.matrix = landscape
        self.dim = dim

        return self.matrix


    def plot(self):
        """Plot the persistence landscape
        """        
        pl.figure()
        for row in self.matrix:
            pl.title(f"$H_{self.dim}$")
            pl.plot(self.grid_m, row)




        














 
