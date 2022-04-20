
import numpy as np
import matplotlib.pyplot as pl

class PersistenceLandscape:

    def __init__(self):
        pass

    # For statistics to make sense, max_m need to be set beforehand.
    def fit(self, pairs, dim = 1, resolution = 100, max_m = None):
        """Find the persistence landscape for a homology dimension

        Parameters
        ----------
        pairs : _type_
            _description_
        dim : int, optional
            _description_, by default 1
        resolution : int, optional
            _description_, by default 100
        max_m : float, optional
            Maximum value for the landscape function. Used to find the grid resolution, 
            if the values are used for statistics or comparison, these should be set to
            a shared value for all the landscapes.
            
        Returns
        -------
        matrix
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
        if max_m is None: 
            max_m = np.max(m) + np.max(h)
        grid_m = np.linspace(0, max_m, resolution)
        lambda_mat = np.zeros((npoints, resolution))

        # Triangular function for each point 
        for i, (u_m, u_h) in enumerate(zip(m, h)):
            lambda_mat[i] = np.maximum( u_h - np.abs(u_m - grid_m), 0)
        # Sort triangles to get λ(m,h)
        landscape = -np.sort(-lambda_mat, axis=0)  

        self.grid_m = grid_m
        self.landscape = landscape
        self.dim = dim

        return self.landscape


    def plot(self):
        pl.figure()
        for row in self.landscape:
            pl.title(f"$H_{self.dim}")
            pl.plot(self.grid_m, row)




        














 
