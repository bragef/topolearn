import networkx as nx
import numpy as np
from scipy import sparse
from itertools import combinations
from .homology import reduce_matrix_bit, reduce_matrix,  find_birth_death_pairs

# Container class for the simplical complexes
# Init with a simplex_collection dictionary:
#    keys: frozenset({nodes})
#    values: tuple(counter, added_idx, filtration_value, distance_value )
class SimplicalComplex:
    def __init__(self, simplex_collection):
        self.simplex_collection = simplex_collection

    # Return the 1-skeleton of the simplex as a
    # networkx graph, add coordinates as w attributes if X is given.
    def graph(self, X=None):
        graph = nx.Graph()
        w = None
        for simplex_set, (idx, dim, fvalue, dvalue) in self.simplex_collection.items():
            simplex = tuple(simplex_set)  # Make subscribtable
            if dim == 0:
                if X is not None:
                    w = X[simplex[0]]
                graph.add_node(simplex[0], w=w, i=idx, f=fvalue, d=dvalue)
            elif dim == 1:
                graph.add_edge(simplex[0], simplex[1])

        return graph



    # Return simplex list indexed by index-number
    # Useful to reclaim simplex from index in boundary matrix
    def as_list(self):
        simplex_list = [
            [dim, fvalue, simplex, dvalue]
            for simplex, (idx, dim, fvalue, dvalue) in self.simplex_collection.items()
        ]
        return simplex_list

    # Create the boundary matrix from the simplices
    def boundary_matrix(self, sparse_mat=False):
        size = len(self.simplex_collection)  # Number of simplices
        bmatrix = np.zeros((size, size), dtype=bool)
        for simplex_set, (idx, dim, fvalue, dvalue) in self.simplex_collection.items():
            if dim >= 1:
                for boundary in combinations(simplex_set, dim):
                    # Get the index number of the faces of the simplex
                    face = frozenset(boundary)
                    bmatrix[self.simplex_collection[face][0], idx] = True
        if sparse_mat:
            return sparse.csc_array(bmatrix)
        else:
            return bmatrix

    # Find the birth death-pairs and add dimension and filtration value to
    # output
    def birth_death_pairs(self, dim=None):
        simplices = self.as_list()  # Simplices indexed by number
        boundaries = self.boundary_matrix()
        reduced_matrix = reduce_matrix_bit(boundaries)
        pairs = find_birth_death_pairs(reduced_matrix)
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


class HomologyPairs:
    def __init__(self, pairs):
        self.pairs = pairs






