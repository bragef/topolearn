import numpy as np
from bitarray import bitarray
from bitarray.util import rindex, zeros


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


# Reduce the boundary matrix
# Same as above, but faster version using bitarrays
def reduce_matrix_bit(boundary_matrix):
    dim = boundary_matrix.shape[0]
    # Initial low-values for matrix. For reduced columns, we set
    # low(B_i) = -1, otherwise low(B_i) = max_j{j: B_ij != 0}
    low = np.array([np.max(np.where(col), initial=-1) for col in boundary_matrix.T])
    r_array = [bitarray(col.tolist()) for col in boundary_matrix.T]
    # Main algorithm
    for j in range(0, dim):
        while True:
            if low[j] == -1:  # Column fully reduced
                break
            # Not ideal - np.where is not very fast  
            [cols] = np.where(low[0:j] == low[j])
            if len(cols) == 0:
                break  # No columns left to add
            i = cols[0]
            # Add the columns mod 2
            r_array[j] = r_array[j] ^ r_array[i]
            low[j] = -1 if not r_array[j].any() else rindex(r_array[j])
    # Convert back to numpy array
    r_matrix = np.zeros((dim,dim), dtype=bool)
    for j, ba in enumerate(r_array):
        r_matrix[:,j] = np.array(ba.tolist(), dtype=bool)
    return r_matrix


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
                # 
                birth_death_pairs.append((j, col[0]))
    return birth_death_pairs



# Reduce the boundary matrix
# Same as above, but faster version using bitarrays
# Improved version using column clearing as described in
# https://boole.cs.qc.cuny.edu/cchen/publications/chen_eurocg_2011.pdf
def reduce_matrix_twist(boundary_matrix):
    dim = boundary_matrix.shape[0]
    # Initial low-values for matrix. For reduced columns, we set
    # low(B_i) = -1, otherwise low(B_i) = max_j{j: B_ij != 0}
    low = np.array([np.max(np.where(col), initial=-1) for col in boundary_matrix.T])
    # For twist algorithm
    simplex_d = np.sum(boundary_matrix,axis=0)
    simplex_d_max = np.max(simplex_d)
    r_array = [bitarray(col.tolist()) for col in boundary_matrix.T]
    # Main algorithm
    for d in range(simplex_d_max,1,-1):
        for j in np.where(simplex_d==d)[0]:
            while True:
                if low[j] == -1:  # Column fully reduced
                    break
                [cols] = np.where(low[0:j] == low[j])
                if len(cols) == 0:
                    break  # No columns left to add
                i = cols[0]
                # Add the columns mod 2
                r_array[j] = r_array[j] ^ r_array[i]
                # r_array[j] ^= r_array[i]
                if r_array[j].any():
                    low[j] = rindex(r_array[j])
                    # Clear the column 
                    low[low[j]] =-1
                    #  r_array[low[j]].setall(0)
                    r_array[low[low[j]]] = zeros(dim)
                else:
                    low[j] = -1 
    # Convert back to numpy array
    r_matrix = np.zeros((dim,dim), dtype=bool)
    for j, ba in enumerate(r_array):
        r_matrix[:,j] = np.array(ba.tolist(), dtype=bool)
    return r_matrix

# Reduce the boundary matrix
# Same as above, but faster version using bitarrays
def update_low(low, low_col):    
    for k in range(0,low.shape[0]):
        lowest_idx = np.where(np.array(low) == k)[0]
        if len(lowest_idx) > 0:
            low_col[k] = lowest_idx[0]
        else:
            low_col[k] = -1
    return low_col


def reduce_matrix_bit_s(boundary_matrix):
    dim = boundary_matrix.shape[0]
    # Initial low-values for matrix. For reduced columns, we set
    # low(B_i) = -1, otherwise low(B_i) = max_j{j: B_ij != 0}
    low = np.array([np.max(np.where(col), initial=-1) for col in boundary_matrix.T])
    low_col = np.ones(dim, dtype=int)
    # Save the lowest index because np.where is quite costly

    r_array = [bitarray(col.tolist()) for col in boundary_matrix.T]    
    # Main algorithm
    for j in range(0, dim):
        low_col = update_low(low, low_col)
        while True:
            if low[j] == -1:  
                # Column fully reduced
                break
            # Lowest column number with current j
            i = low_col[low[j]]
            if i == -1 or i >= j:
                # Done! 
                break            
            # Add the columns mod 2
            r_array[j] = r_array[j] ^ r_array[i]
            low[j] = -1 if not r_array[j].any() else rindex(r_array[j])
    # Convert back to numpy array
    r_matrix = np.zeros((dim,dim), dtype=bool)
    for j, ba in enumerate(r_array):
        r_matrix[:,j] = np.array(ba.tolist(), dtype=bool)
    return r_matrix

