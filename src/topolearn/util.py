# No package is complete without a util.py

import matplotlib.pyplot as pl
import matplotlib.colors as colors
import numpy as np
import networkx as nx
import pandas as pd

# Plot graph together with data points for example plots.
# X should be 2D data, all nodes should have a w attribute with position
def plot_graph_with_data(graph, X, axis=False, alpha=0.25):
    # graph = learner.fit(X)
    fig, ax = pl.subplots()
    pos = {n: (d["w"][0], d["w"][1]) for (n, d) in graph.nodes(data=True)}
    nx.draw(graph, pos, node_color="r", edge_color="b", alpha=alpha, node_size=2, ax=ax)
    ax.scatter(X[:, 0], X[:, 1], s=1, alpha=0.15)
    if axis:
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)


def plot_persistance_diagram(pairs, max_dim=None, show_infinite=True, size=20, size_diagonal=0.1):
    # Max dimension never die, remove from plot.
    pairs = np.array(pairs)
    if max_dim is None:
        max_dim = np.nanmax(pairs[:, 2])
    incl = np.where(pairs[:, 2] <= max_dim)
    [d] = np.array(pairs[incl, 4], dtype=float)
    [b] = np.array(pairs[incl, 3], dtype=float)
    [dim] = pairs[incl, 2]
    # Plot the birth-death pairs as circles
    dimcolours = ["red", "green", "blue", "purple"]
    pl.figure()
    # Ephemeral cycles which disappear within the same filtration values
    # Plot these as small dots, the non-ephemeral as larger circles
    is_noise = (b - d) == 0
    s = np.ones(len(dim), dtype=float)  # Marker sizes
    s[is_noise] = size_diagonal
    s[is_noise == False] = size
    pl.scatter(
        b, d, c=dim, cmap=colors.ListedColormap(dimcolours), alpha=0.3, marker="o", s=s
    )
    # And the infinite pairs as triangles
    if show_infinite:
        undead = np.where(np.isnan(d))
        maxd = np.nanmax(d)
        pl.scatter(
            b[undead],
            maxd * np.ones_like(undead),
            c=dim[undead],
            cmap=colors.ListedColormap(dimcolours),
            alpha=0.5,
            marker="^",
        )
    


# Two nested balls with uniformily distributed points on the surface.
# The noise parameter controls the radius. ~ 3d version of make_circles 
def make_shells(n, dim=3, noise=0):
    X = np.random.normal(size=n * dim)
    X = np.reshape(X, (n, dim))
    X /= np.reshape(np.linalg.norm(X, axis=1), (n, 1))
    y = np.random.choice([1, 3], n, p=[1 / 4, 3 / 4])
    if noise > 0:
        X *= np.reshape(
            y * np.random.normal(loc=1, scale=noise, size=n),
            (n, 1),
        )
    else:
        X *= np.reshape(y, (n, 1))
    return (y, X)

