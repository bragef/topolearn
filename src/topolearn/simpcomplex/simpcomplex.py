import networkx as nx
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.colors as colors
from scipy import sparse
from itertools import combinations
from ..persistence import reduce_matrix_set, find_birth_death_pairs_set

# Container class for the simplical complexes
# Init with a simplex_collection dictionary:
#    keys: frozenset({nodes})
#    values: tuple(counter, added_idx, filtration_value, distance_value )
class SimplicalComplex:
    def __init__(self, simplex_collection):
        self.simplex_collection = simplex_collection
        self.simplex_index = None       # Use lazy construction 

    # Return the 1-skeleton of the simplex as a
    # networkx graph, add coordinates as w attributes if X is given.
    def graph(self, X=None):
        graph = nx.Graph()
        w = None
        for simplex_set, (idx, dim, fvalue) in self.simplex_collection.items():
            simplex = tuple(simplex_set)  # Make subscribtable
            if dim == 0:
                if X is not None:
                    w = X[simplex[0]]
                graph.add_node(simplex[0], w=w, i=idx, f=fvalue)
            elif dim == 1:
                graph.add_edge(simplex[0], simplex[1])

        return graph

    # Return simplex list indexed by index-number
    # Useful to reclaim simplex from index in boundary matrix 
    # Applications should use the get_simplex(idx) method
    def as_list(self):
        simplex_list = [
            [dim, fvalue, simplex]
            for simplex, (idx, dim, fvalue) in self.simplex_collection.items()
        ]
        return simplex_list

    # Retrieve a simplex from its index value.
    # Returns a tuple (dim, birth_value, simplex, death_value)
    def get_simplex(self, idx):
        if self.simplex_index is None:  
            self.simplex_index = self.as_list()
        return self.simplex_index[idx]


    # Create the boundary matrix from the simplices
    def boundary_matrix(self, sparse_mat=False):
        size = len(self.simplex_collection)  # Number of simplices
        bmatrix = np.zeros((size, size), dtype=bool)
        for simplex_set, (idx, dim, fvalue) in self.simplex_collection.items():
            if dim >= 1:
                for boundary in combinations(simplex_set, dim):
                    # Get the index number of the faces of the simplex
                    face = frozenset(boundary)
                    bmatrix[self.simplex_collection[face][0], idx] = True
        if sparse_mat:
            return sparse.csc_array(bmatrix)
        else:
            return bmatrix

    # Create the boundary matrix as list of sets from the simplices
    # The return value is a list of sets, where the sets contains the index value 
    # of the non-zero entries in the boundary matrix for each column.
    def boundary_sets(self):
        size = len(self.simplex_collection)  # Number of simplices
        boundary_cols = [ set() for i in range(0, size)]
        for simplex_set, (idx, dim, fvalue) in self.simplex_collection.items():
            if dim >= 1:
                for boundary in combinations(simplex_set, dim):
                    # Get the index number of the faces of the simplex
                    try:
                        face = frozenset(boundary)
                        boundary_cols[idx] |= { self.simplex_collection[face][0] }
                    except:
                        print(f"Error: face {simplex_set} has missing boundary ({face})!")
                        return None

        return boundary_cols
    

    # Find the birth death-pairs and add dimension and filtration value to
    # output
    def birth_death_pairs(self, dim=None, verbose=1):

        simplices = self.as_list()  # Simplices indexed by number
        boundaries = self.boundary_sets()
        reduced_matrix = reduce_matrix_set(boundaries, verbose=verbose)
        pairs = find_birth_death_pairs_set(reduced_matrix)
        pairs_out = list()
        for (b, d) in pairs:
            # Add dimension and filtration values
            sdim = simplices[b][0]
            if dim is not None and sdim != dim:
                continue
            birth_f = simplices[b][1]  # Filtration value at birth
            death_f = simplices[d][1] if d is not None else None  # ..and death
            pairs_out.append((b, d, sdim, birth_f, death_f))
        return pairs_out

#  We want to compare two sets of birth-death simplices with the assosicated death
# distance
# 
# { {a}, {b} }
# def birth_death_sets(simplices, bdpairs):



class Persistence:

    def __init__(self, pairs):
        self.pairs = pairs

    def plot(self,  max_dim=None, show_infinite=True, size=20, size_diagonal=0.1):
        # Max dimension never die, remove from plot.
        pairs = np.array(self.pairs)
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
    
    # Return a dictionary of (edge) => (persistance)
    # Only handles 0 and 1 homologies yet
    def as_dict(self, dim):
        out = {}
        for p in self.pairs:
            if p[1] is not None  and p[2] == dim:
                out[ frozenset({p[0], p[1]})] = p[4]
        return out






