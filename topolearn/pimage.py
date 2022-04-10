import numpy as np
from scipy.stats import norm

# A limited implementation of Persistence Images, as described in Adams et.al 2017:
# Persistence Images: A Stable Vector Representation of Persistent Homology
#

# Default weight function suggested by Adams et.al.
def weight_linear(u_p, p_max):
    if u_p < 0:
        return 0
    elif u_p > 1:
        return 1
    else:
        return u_p / p_max

# 
class PersistenceImage:
    def __init__(self):
        pass

    # p_max = max persistence for weight function
    # weight_fun = function which takes persistance and max(persistance) 
    def fit(self, pairs, sigma=1, resolution=50, p_max=None, b_max=None, weight_fun = weight_linear):
        # Transform into birth-persistance coordinates
        pairs = np.array(pairs)
        dim = np.array(pairs[:, 2], dtype=int) 
        b = np.array(pairs[:, 3], dtype=float)  # Births
        d = np.array(pairs[:, 4], dtype=float)  # Deaths
        # Persistance values, T(B) = (b, d-b)
        # T(B) = (b, d-b)

        # Calculate grid and max birth/persistane and create a common
        # grid for all dimensions
        p = d - b
        if p_max is None:
            p_max = np.nanmax(p)
        if b_max is None:
            b_max = np.nanmax(b)
        # grid_max = max(b_max, p_max)    # Symmetric grid? 
        p_grid = np.linspace(0, p_max, num=resolution)
        delta_p = p_max / resolution
        b_grid = np.linspace(0, b_max, num=resolution)
        delta_b = b_max / resolution
        # Allocate images for each dimension
        images = np.zeros(( np.max(dim), resolution, resolution ))

        # We only need the pairs with non-zero and finite persistance
        valid = np.where(np.isfinite(p) & (p > 0))
        # ρ(z) = Σ w(u) φ_u(z)
        # φ is a Gaussian pdf with diagonal covariance, therefore
        # g(x,y)=g(x)g(y), and I(x,y) = Σ_u w(u_p) ∫g(x,u_b)∫g(y,u_p))
        for u_b, u_p, sdim in zip(b[valid], p[valid], dim[valid]):
            images[sdim] += np.outer(
                weight_fun(u_p, p_max)
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
        self.extent = [0, b_max, 0, p_max  ]
        # Midpoint grid for plotting
        self.b_grid = b_grid + delta_b / 2
        self.p_grid = p_grid + delta_p / 2
        # Max homology dimenshion
        self.max_dim = images.shape[0]-1      
        return self.images

    #  ¯\_(ツ)_/¯
    def as_vector(self):
        return np.reshape(self.images, -1)




