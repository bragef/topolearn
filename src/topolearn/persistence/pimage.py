import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import norm


# A limited implementation of Persistence Images, as described in Adams et.al 2017:
# Persistence Images: A Stable Vector Representation of Persistent Homology


# Default weight function suggested by Adams et.al.
def weight_linear(u_p, p_max):
    """Linear weight function"""
    if u_p < 0:
        return 0
    elif u_p > 1:
        return 1
    else:
        return u_p / p_max


#
class PersistenceImage:
    """Persistence images

    Calculate the persistence images for all homology dimensions av a persistence pair
    set.

    Parameters
    ----------
    sigma : int, optional
        Standard deviation for the Gaussian kernel
    resolution : int, optional
        Grid resolution
    p_max, p_max : float, optional
        Maximum birth and death for output. If not specifed, max values of
        data will be used.
    weight_fun : function(p, p_max) -> [0,1], optional
        Weigth function for persistence values. If not specified, a linear weight 
        will be used.

    Examples
    --------
    >>> from topolearn.simpcomplex import AlphaComplex
    >>> from topolearn.persistence import PersistenceLandscape
    >>> from sklearn.datasets import make_circles

    >>> X, _ = make_circles(noise=0.1,  n_samples=500, random_state=50)
    >>> learner = AlphaComplex()

    >>> simplices = learner.fit(X)
    >>> pairs = simplices.birth_death_pairs()
    >>> p = PersistenceImage(resolution=100, sigma=0.1)
    >>> p.fit(pairs)
    >>> p.plot()

    Notes
    -----
    Persistence images are described in 
    Adams, Henry, Tegan Emerson, Michael Kirby, Rachel Neville, Chris Peterson, 
    Patrick Shipman, Sofya Chepushtanova, Eric Hanson, Francis Motta, and Lori 
    Ziegelmeier. 2017. ‘Persistence Images: A Stable Vector Representation of 
    Persistent Homology’. Journal of Machine Learning Research 18 (8): 1–35.
    """


    def __init__(
        self, sigma=1, resolution=50, p_max=None, b_max=None, weight_fun=weight_linear
    ): 
        self.sigma = sigma
        self.resolution = resolution
        self.p_max = p_max
        self.b_max = b_max
        self.weight_fun = weight_fun

    # p_max = max persistence for weight function
    # weight_fun = function which takes persistance and max(persistance)
    def fit(self, pairs):
        """Find the persistence images for a persistence pair set

        Parameters
        ----------
        pairs : array
            List of birth-death pairs as returned by SimplicalComplex.birth_death_pairs)
            
        Returns
        -------
        (array, shape=(resoution, n_persist))
            Matrix with persistence landscape values
   
        """

        # Transform into birth-persistance coordinates
        pairs = np.array(pairs)
        dim = np.array(pairs[:, 2], dtype=int)
        b = np.array(pairs[:, 3], dtype=float)  # Births
        d = np.array(pairs[:, 4], dtype=float)  # Deaths
        # Persistance values, T(B) = (b, d-b)
        # T(B) = (b, d-b)

        # Calculate grid and max birth/persistane and create a common
        # grid for all dimensions
        p_max = self.p_max
        b_max = self.b_max
        p = d - b
        if p_max is None:
            p_max = np.nanmax(p)
        if b_max is None:
            b_max = np.nanmax(b)
        # grid_max = max(b_max, p_max)    # Symmetric grid?
        p_grid = np.linspace(0, p_max, num=self.resolution)
        delta_p = p_max / self.resolution
        b_grid = np.linspace(0, b_max, num=self.resolution)
        delta_b = b_max / self.resolution
        # Allocate images for each dimension
        images = np.zeros((np.max(dim) + 1, self.resolution, self.resolution))

        # We only need the pairs with non-zero and finite persistance
        valid = np.where(np.isfinite(p) & (p > 0))
        self.max_dim = images.shape[0]

        # ρ(z) = Σ w(u) φ_u(z)
        # φ is a Gaussian pdf with diagonal covariance, therefore
        # g(x,y)=g(x)g(y), and I(x,y) = Σ_u w(u_p) ∫g(x,u_b)∫g(y,u_p))
        sigma = self.sigma
        for u_b, u_p, sdim in zip(b[valid], p[valid], dim[valid]):
            images[sdim] += np.outer(
                self.weight_fun(u_p, p_max)
                * (
                    norm.cdf(p_grid + delta_p, loc=u_p, scale=sigma)
                    - norm.cdf(p_grid, loc=u_p, scale=sigma)
                ),
                (
                    norm.cdf(b_grid + delta_b, loc=u_b, scale=sigma)
                    - norm.cdf(b_grid, loc=u_b, scale=sigma)
                ),
            )

        self.images = images
        self.extent = [0, b_max, 0, p_max]
        # Midpoint grid for plotting
        self.b_grid = b_grid + delta_b / 2
        self.p_grid = p_grid + delta_p / 2

        # If last dimension have no data, remove
        self.max_dim = np.max(dim[valid])

        return self.images

    def plot(self):
        """Plot the persistence images for all homology dimensions
        """       
        for i in range(0, self.max_dim + 1):
            pl.figure()
            pl.title(f"H$_{i}$")
            pl.imshow(self.images[i], origin="lower", extent=self.extent, cmap="Blues")

    #  ¯\_(ツ)_/¯
    def as_vector(self):
        return np.reshape(self.images, -1)
