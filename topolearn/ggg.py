
import numpy as np
import networkx as nx
import scipy.stats as stats
from scipy.special import erf
import random
from scipy.spatial import Delaunay
from sklearn import mixture


#  Algorithm. 
#
# 1. Initialisation of points using standard algorithm, GNN, or KNN
#   1.1. Initialisation of k points
#   1.2. Standard GMM with k centers
#   1.3. Delaney graph
#   1.4. Initialise mixture model with equal weights for all edges and vertices
# 2. Optimize likelihood using EM-algorithm
#   2.1 Update the π-vector (M-step)
#       and shared variance σ^2 (E-step), 
#       (Aupetit (4) and (5))
#   2.3 Repeat EM until convergence or t_max is 
# 3. Prune edges with weigths below 
#

class GGG():

    def __init__(
        self,
        eps=0.1,
        k=20,
        max_iter = 10,
        sigma = 10
    ):
        self.eps = eps            # Pruning threshold
        self.k = k                # Number of component
        self.GG = None            # Graph representation
        self.D = None             # Data dimension
        self.max_iter = max_iter  #
        self.sigma = sigma
        self.means_ = None
        self.mprobs = None

    def fit(self, X, y = None):

        #X = X * 100
        self.D = D = X.shape[1]  

        # Initalisation with GMM and random points
        w_init = self.fit_gmm(X, self.k)
        # Delauney triangulation
        w_delauney = Delaunay(w_init)
     
        # Delauney graph to networkx.graph
        DG = nx.Graph()
        for nodeid, w in enumerate(w_delauney.points):
            DG.add_node(nodeid, w=w)
        # Simplices to unique edges
        for path in w_delauney.simplices:
            nx.add_cycle(DG, path)

        N0 = DG.number_of_nodes()
        N1 = DG.number_of_edges()
        # Initalise mixture probabilities
        pi = [ np.ones((N0,1)), np.ones((N1,1)) ]
        pi[0] *= 1/(N0 + N1)
        pi[1] *= 1/(N0 + N1)
        # No prior info on sigma, initialise with 1
        sigma = self.sigma
        # Make the distance matrices:
        (self.Q, self.d_edge, self.d_vertex, self.L) = \
             calc_distances(X, DG)

        # Run EM algorithm 
        # for i in range(0, self.max_iter):
        for i in range(0,self.max_iter):
            if sigma < np.finfo(float).eps:
                print("Did not converge, sigma = 0")
                break
            print(f"Iteration {i}, sigma={sigma}")
            (pi, sigma) = self.step_update(pi, sigma)

        print(nodeid)
        return DG

    def fit_gmm(self, X, n_components):
        # It would make sense to experiment with initialisations. 
        # 'tied' covariance make some sense here, since the
        # GGG model a shared variance coponent.
        gmm = mixture.GaussianMixture(
                n_components=n_components, 
                covariance_type='tied',
                init_params='kmeans'           
                )
        gmm.fit(X)    
        return gmm.means_

    # E and M-step.
    # 
    def step_update(self, pi, sigma):
        
        # Number of data points
        M = self.Q.shape[0]
        # Treat values below this value as zeros
        zero = np.finfo(float).eps
        # Distributions. 
        # g0: M x N0, g1: M x N1 
        g0 = g0_matrix(self.d_vertex, sigma, self.D)
        g1 = g1_matrix(self.Q, self.d_edge, self.L, sigma, self.D)
        # Pointwise probabilties 
        p0 = g0 * np.transpose(pi[0]) 
        p1 = g1 * np.transpose(pi[1])

        # Normalisation constants for v_j: M x 1 
        # Aupetit Eq. (2):
        pvj = np.reshape( np.sum(p0, axis=1) + np.sum(p1, axis=1), (M, 1))
        pvj_inv = np.divide(1, pvj, out=np.zeros_like(pvj), where = pvj > zero)
        # Update mixture probabilities, 
        # Aupetit Eq. (4), pi
        pi[0] = np.sum( p0 * pvj_inv , axis=0) / M
        pi[1] = np.sum( p1 * pvj_inv , axis=0) / M
        print(np.sum(pi))
        # debug
        p0 = g0 * np.transpose(pi[0]) 
        p1 = g1 * np.transpose(pi[1])
        # Update sigma 
        Lt = np.transpose(self.L)
        # Aupetit Eq. (5) 
        I1 = sigma*np.sqrt(np.pi / 2) * \
             (erf(self.Q / (sigma * np.sqrt(2))) -
              erf((self.Q - Lt)  / (sigma * np.sqrt(2))))
        I2 = sigma**2  * \
            ( (self.Q - Lt) * np.exp(-( self.Q - Lt )**2/(2*sigma**2)) - \
                self.Q * np.exp( - self.Q**2/( 2*  sigma**2) ))
        # Aupetit Eq. (4), sigma
        # The zeros have no contribution to sigma, set to 0
        p1g1 = np.divide(p1, g1, out=np.zeros_like(g1), where = g1 > zero)

        sigma_vertices = np.sum(p0 * self.d_vertex**2) 

        _tmp = p1g1 * np.power(2*np.pi*sigma**2, - self.D/2) 
        _tmp *= np.exp( - self.d_edge**2 / ( 2 * sigma**2) ) 
        _tmp *= ( I1 * ( self.d_edge**2 + sigma**2) + I2 ) 
        sigma_edges = np.sum(_tmp)

        sigma = np.sqrt( (sigma_edges + sigma_vertices) / (self.D * M) )
        sigma = np.sqrt( (sigma_edges + sigma_vertices) )

        return((pi, sigma))

# Calculate distance matrices:
#  
# Q_ij length along the vertex
# Lvq distance from v_i to projection on the vertex. 
# The likelihood formulas only use the distance from the vertex,
# so we do not need to store the individual q_ij's, so the matrix 
# In addition to the Qij (M x N1) matrix, calculate
# Lvq = || q_ij - v_i || (M x N1)
# Leq = || w - v_j ||  ( M x N0 )
# L_vec = || w_a - w_b ||  ( N1 x 1)
def calc_distances(X, DG):
    # Allocate matrices with dimension n x N1
    Q_matrix = np.zeros((len(X), DG.number_of_edges()))
    # Save || v_j - q_j || 
    d_edge = np.zeros((len(X), DG.number_of_edges()))
    d_vertex = np.zeros((len(X), DG.number_of_nodes()))
    L_vec = np.zeros(DG.number_of_edges())

    # Length of each edge
    for j, (n1, n2) in enumerate(DG.edges()):
        L_vec[j] =  np.linalg.norm(DG.nodes[n1]["w"] - DG.nodes[n2]["w"]) 
    # Reshape L_vec from (n,) to (n,1)
    # This will make elementwise operation between Q (dim n x k ) and L (dim k)
    # understood as a rowwise operation  
    L_vec = L_vec[:, np.newaxis]

    for i, v  in enumerate(X):
        for j, (n1, n2, attrib) in enumerate(DG.edges(data=True)):
            w_a = DG.nodes[n1]['w']
            w_b = DG.nodes[n2]['w']

            Q_ij = np.dot(v - w_a, w_b - w_a) / L_vec[j]

            q_ij =  w_a + (w_b - w_a) * Q_ij / L_vec[j]
            Q_matrix[i, j] =  Q_ij
            d_edge[i, j] = np.linalg.norm(v - q_ij)

    for j, (v, attrib) in enumerate(DG.nodes(data=True)):
        d_vertex[:,j] = np.linalg.norm(X - attrib['w'], axis=1)
    
    return((Q_matrix, d_edge, d_vertex, L_vec ))


# g0: Vertex probabilties.
# 
# Input is a matrix of distance from point to node, and current value
# og the sigma parameter.
# 
# Return a matrix of g0 probabilties for all points in the sample. Use the
# pre-centered (vi-qi) values.
#
# Probability here is really just from multivariate normal
# with Σ = σ I_D. Since we have precalculated the norms here, we
# just use scikit.stats.norm.pdf + adjustment for the number of dimensions.
# Output dimension: M x N0
def g0_matrix(Leq, sigma, D):
    return (np.power(2*np.pi*sigma**2,-D/2) * \
        np.exp(- Leq **2 / ( 2*sigma**2) ))

# g1: Edge probabilities
# 
# Lots of calculations here!
# Calculate g1 for all edges.
# Input dimensions: Q_ij (M x N1), Lvq (N1), L_vec (1 x N1) 
# Output dimension: M x N1
def g1_matrix(Q, d_edge, l_edge, sigma, D):
    # Edgewise operations, want rows as edges

    Q_ji = np.transpose(Q)    
    # Eq.(1) Aupetit 2005
    res = np.power(2*np.pi*sigma**2,-(D-1)/2) 
    res *=  np.exp(- np.transpose(d_edge)**2 / ( 2*sigma**2) )  
    res *=  1/(2 * l_edge) 
    res *=  (erf(Q_ji / (sigma * np.sqrt(2)))  - \
                 erf(( Q_ji - l_edge)  / (sigma * np.sqrt(2))))
    # Return matrix with data as rows, edges as columns
    return np.transpose(res)


# def Pv0(distances, pi):

