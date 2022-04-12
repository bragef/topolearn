import numpy as np
from time import time


def reduce_matrix_set(boundary_matrix):
    # Represent columns as a set of non-zero simplices
    # r_array =  [ set(np.where(col)[0]) for col in boundary_matrix.T ]
    r_array = boundary_matrix
    dim = len(r_array)
    # dim = boundary_matrix.shape[0]
    # Initial low-values for matrix. For reduced columns, we set
    # low(B_i) = -1, otherwise low(B_i) = max_j{j: B_ij != 0}
    low = np.array([ max(b) if len(b)>0  else -1 for b in r_array])
    # Main algorithm
    steps = 0
    t_start = time()
    for j in range(0, dim):
        while True:
            steps += 1
            if low[j] == -1:  # Col fully reduced
                break
            [cols] = np.where(low[0:j] == low[j])
            if len(cols) == 0:
                break  # No columns left to add
            i = cols[0]
            r_array[j] ^= r_array[i]
            low[j] = max(r_array[j]) if len(r_array[j])>0 else -1
    print(f'Reduced matrix in {steps} steps in {round(time() - t_start, 2)} sec.')        
    return r_array

def find_birth_death_pairs_set(reduced_set):
    low = np.array([ max(b) if len(b)>0  else -1 for b in reduced_set]) 
    birth_death_pairs = list()
    for j, i in enumerate(low):
        if i == -1:  # Birth
            [col] = np.where(low == j)  # 
            if len(col) == 0:  # No death, None for infinity
                birth_death_pairs.append((j, None))
            else:
                birth_death_pairs.append((j, col[0]))
    return birth_death_pairs



## Ignore the functions below this line.

# from bitarray import bitarray
# from bitarray.util import rindex, zeros



# Reduce the boundary matrix, standard algort
# (Should probably try a sparse matrix class here?)
def reduce_matrix(boundary_matrix):
    # Passed by ref; make a copy
    reduced_matrix = boundary_matrix.copy()
    dim = reduced_matrix.shape[0]
    # Initial low-values for matrix. For reduced columns, we set
    # low(B_i) = -1, otherwise low(B_i) = max_j{j: B_ij != 0}
    low = np.array([np.max(np.where(col), initial=-1) for col in reduced_matrix.T])
    # Main algorithm
    for j in range(0, dim):
        while True:
            if low[j] == -1:  # Col fully reduced
                break
            [cols] = np.where(low[0:j] == low[j])
            if len(cols) == 0:
                break  # No columns left to add
            i = cols[0]
            # Add the columns mod 2
            reduced_matrix[:, j] = np.logical_xor(
                reduced_matrix[:, j], reduced_matrix[:, i]
            )
            # Update the low function
            low[j] = np.max(np.where(reduced_matrix[:, j]), initial=-1)
    return reduced_matrix



# Given the reduced matrix, return the birth-death pairs.
# Returns a list of birt-death value. Death set to none
# for infinite pairs
def find_birth_death_pairs(reduced_matrix):
    low = np.array([np.max(np.where(col), initial=-1) for col in reduced_matrix.T])
    birth_death_pairs = list()
    for j, i in enumerate(low):
        if i == -1:  # Birth
            [col] = np.where(low == j)  # 
            if len(col) == 0:  # No death, None for infinity
                birth_death_pairs.append((j, None))
            else:
                birth_death_pairs.append((j, col[0]))
    return birth_death_pairs


# Reduce the boundary matrix
# Same as above, but faster version using bitarrays
def reduce_matrix_bit(boundary_matrix):
    dim = boundary_matrix.shape[0]
    # Initial low-values for matrix. For reduced columns, we set
    # low(B_i) = -1, otherwise low(B_i) = max_j{j: B_ij != 0}
    low = np.array([np.max(np.where(col), initial=-1) for col in boundary_matrix.T])
    r_array = []
    for col in boundary_matrix.T:
        c=bitarray()
        c.pack(col.tobytes())
        r_array.append(c)
    ops = 0
    # Main algorithm
    for j in range(0, dim):
        while True:
            ops += 1
            if low[j] == -1:  # Column fully reduced
                break
            # Not ideal - np.where is not very fast  
            [cols] = np.where(low[0:j] == low[j])
            if len(cols) == 0:
                break  # No columns left to add
            i = cols[0]
            # Add the columns mod 2
            r_array[j] ^= r_array[i]
            low[j] = -1 if not r_array[j].any() else rindex(r_array[j])
    # Convert back to numpy array
    r_matrix = np.zeros((dim,dim), dtype=bool)
    for j, ba in enumerate(r_array):
        r_matrix[:,j] = np.frombuffer(ba.unpack(), dtype=bool)
    print(f'Reduced matrix in {ops} steps')
    return r_matrix