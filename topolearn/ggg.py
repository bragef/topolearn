import numpy as np
import networkx as nx
import scipy.stats as stats
from scipy.special import erf
from scipy.spatial import Delaunay
from sklearn import mixture
from sklearn.cluster import MiniBatchKMeans


#  Algorithm from Aupetit 2005:
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
# 

class GenerativeGaussianGraph:
    def __init__(self, eps=0.1, k=20, max_iter=10, sigma=10, init_method="GNN"):
        self.eps = eps  # Pruning threshold
        self.k = k  # Number of component
        self.D = None  # Data dimension
        self.max_iter = max_iter  #
        self.init_method = init_method
        # Output
        self.sigma = sigma
        self.graph = None  # Graph representation
        self.pi = None
        # Config.
        self.conv_rate = 0.001  # Stop when Δσ/σ < conv_rate


    def fit(self, X, y=None):

        self.D = D = X.shape[1]

        # Init nodes with GMM or KMeans
        w_init = self.fit_init(X, self.k, method=self.init_method)
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
        pi = [np.ones((N0, 1)), np.ones((N1, 1))]
        pi[0] *= 1 / (N0 + N1)
        pi[1] *= 1 / (N0 + N1)
        sigma = self.sigma

        # Calculate the data distance matrices:
        (self.Q, self.d_edge, self.d_vertex, self.L) = calc_distances(X, DG)

        # Run EM algorithm
        for i in range(0, self.max_iter):
            if sigma < np.finfo(float).eps:
                print("Did not converge, sigma = 0")
                break
            prev_sigma = sigma
            (pi, sigma) = self.step_update(pi, sigma)
            print(f"Iteration {i}, sigma={sigma}")
            if (prev_sigma - sigma) / sigma < self.conv_rate:
                print("Converged")
                break

        # Add probability weigths to graph
        for (p, n) in zip(pi[0], DG.nodes):
            DG.nodes[n]["p"] = p
        for (p, (n1, n2)) in zip(pi[1], DG.edges()):
            DG.add_edge(n1, n2, p=p)

        self.sigma = sigma
        self.pi = pi
        self.graph = DG

        return self.graph

    # Initial fit.
    def fit_init(self, X, n_components, method="GMM"):
        if method == "GMM":
            # It would make sense to experiment with initialisations. 
            # We use 'tied' covariance here, which is reasonable since the
            # GGG model use a shared variance coponent. 
            gmm = mixture.GaussianMixture(
                n_components=n_components, covariance_type="tied", init_params="kmeans"
            )
            gmm.fit(X)
            node_centers = gmm.means_
        else:  
            # method=='KMeans':
            # For large data samples, this should be a lot faster than GNN
            # initialisation            
            kmeans = MiniBatchKMeans(n_clusters=n_components, random_state=0).fit(X)
            node_centers = kmeans.cluster_centers_

        return node_centers

    # E and M-step.
    def step_update(self, pi, sigma):

        # Number of data points
        M = self.Q.shape[0]
        # Distributions.
        # g0: M x N0, g1: M x N1
        # Data in rows (axis=0), components in columns (axis=1)
        g0 = g0_matrix(self.d_vertex, sigma, self.D)
        g1 = g1_matrix(self.Q, self.d_edge, self.L, sigma, self.D)

        # assert np.sum(pi[0]) + np.sum(pi[1]) == 1, "Probabilities does not sum to 1"
        # Pointwise probabilties.
        # (Summing over axis=1 here corresponds to Aupetit Eq. (2))
        (p0, p1) = p_matrix(pi, g0, g1)

        # M - step: Update mixture probabilities,
        # Aupetit Eq. (4), pi
        pi[0] = np.sum(p0, axis=0) / M
        pi[1] = np.sum(p1, axis=0) / M

        # Update edge and vertex probabilities with new PI's
        (p0, p1) = p_matrix(pi, g0, g1)

        # E-step: Update sigma using current probabilities
        Lt = np.transpose(self.L)
        # Aupetit Eq. (5)
        I1 = (
            sigma
            * np.sqrt(np.pi / 2)
            * (
                erf(self.Q / (sigma * np.sqrt(2)))
                - erf((self.Q - Lt) / (sigma * np.sqrt(2)))
            )
        )
        I2 = sigma**2 * (
            (self.Q - Lt) * np.exp(-((self.Q - Lt) ** 2) / (2 * sigma**2))
            - self.Q * np.exp(-self.Q**2 / (2 * sigma**2))
        )
        # Aupetit Eq. (4), sigma
        # The zeros have no contribution to sigma, set to 0
        p1g1 = np.divide(p1, g1, out=np.zeros_like(g1), where=g1 > np.finfo(float).eps)
        # p1g1 = np.divide(p1, g1)
        sigma_vertices = p0 * self.d_vertex**2
        sigma_edges = p1g1 * np.power(2 * np.pi * sigma**2, -self.D / 2)
        sigma_edges /= Lt
        sigma_edges *= np.exp(-self.d_edge**2 / (2 * sigma**2))
        sigma_edges *= I1 * (self.d_edge**2 + sigma**2) + I2

        sigma = np.sqrt((np.sum(sigma_edges) + np.sum(sigma_vertices)) / (self.D * M))

        return (pi, sigma)

    # Return a pruned version of the graph.
    # Note that the pruned version keep the original probability weights, not
    # renormalised to the pruned tree.
    def pruned_graph(self, eps=1):
        g = self.graph.copy()
        remove_edges = [(n1, n2) for n1, n2, p in g.edges(data="p") if p < eps]
        print(f"Removed {len(remove_edges)} edges")
        g.remove_edges_from(remove_edges)
        return g

    # Return edge probabilities, useful for knee plots
    def edge_probs(self):
        edges = self.graph.edges(data=True)
        edgeprobs = sorted([ attrib['p'] for (n1, n2, attrib) in edges], reverse=True)
        return edgeprobs


    # Retrieve mixing probabilities (pi) from graph
    def mprobs(self, graph=None):
        if graph is None:
            graph = self.graph
        pi0 = [p for n1, p in graph.nodes(data="p")]
        pi1 = [p for n1, n2, p in graph.edges(data="p")]
        # Renormalise in case of removed components
        pic = np.sum(pi0) + np.sum(pi1)
        pi0 /= pic
        pi1 /= pic
        return (pi0, pi1)

    # Transform data
    # Return a tuple of of (p0, p1),
    def transform(self, X, graph=None):
        if graph is None:
            graph = self.graph
        (Q, d_edge, d_vertex, L) = calc_distances(X, graph)
        # Extract the probability weights from current graph.  Since we may be
        # dealing with a pruned subgraph, we cannot use self.pi
        pi = self.mprobs(graph)
        g0 = g0_matrix(d_vertex, self.sigma, self.D)
        g1 = g1_matrix(Q, d_edge, L, self.sigma, self.D)

        (p0, p1) = p_matrix(pi, g0, g1)
        return (p0, p1)


# Calculate distance matrices:
#
# Q_ij length along the vertex
# Lvq distance from v_i to projection on the vertex.
# The likelihood formulas only use the distance from the vertex,
# so we do not need to store the individual q_ij's.
# In addition to the Qij (M x N1) matrix, calculate
# d_edge = || q_ij - x_i || (M x N1)
# d_vertex = || w - x_j ||  (M x N0)
# L_vec = || w_a - w_b || (N1 x 1)
def calc_distances(X, DG):
    # Allocate matrices with dimension n x N1
    Q_matrix = np.zeros((len(X), DG.number_of_edges()))
    # Save || x_j - q_j ||
    d_edge = np.zeros((len(X), DG.number_of_edges()))
    d_vertex = np.zeros((len(X), DG.number_of_nodes()))
    L_vec = np.zeros(DG.number_of_edges())

    # Length of each edge
    for j, (n1, n2) in enumerate(DG.edges()):
        L_vec[j] = np.linalg.norm(DG.nodes[n1]["w"] - DG.nodes[n2]["w"])
    # Reshape L_vec from (n,) to (n,1)
    # This will make elementwise operation between Q (dim n x k ) and L (dim k)
    # understood as a rowwise operation
    L_vec = np.reshape(L_vec, (-1, 1))
    for i, v in enumerate(X):
        for j, (n1, n2, attrib) in enumerate(DG.edges(data=True)):
            w_a = DG.nodes[n1]["w"]
            w_b = DG.nodes[n2]["w"]
            Q_ij = np.dot(v - w_a, w_b - w_a) / L_vec[j]
            q_ij = w_a + (w_b - w_a) * Q_ij / L_vec[j]
            Q_matrix[i, j] = Q_ij
            d_edge[i, j] = np.linalg.norm(v - q_ij)

    for j, (v, attrib) in enumerate(DG.nodes(data=True)):
        d_vertex[:, j] = np.linalg.norm(X - attrib["w"], axis=1)

    return (Q_matrix, d_edge, d_vertex, L_vec)


# g0: Vertex distribution.
# Input is a matrix of distance from point to node, and current value
# og the sigma parameter.
#
# Return a matrix of g0 probabilties for all points in the sample. Use the
# pre-centered (xi-qi) values.
#
# Multivariate normal pdf.
# Output dimension: M x N0
def g0_matrix(d_vertex, sigma, D):
    return np.power(2 * np.pi * sigma**2, -D / 2) * np.exp(
        -(d_vertex**2) / (2 * sigma**2)
    )


# g1: Edge distribution
# Calculate g1 for all edges.
# Input dimensions: Q_ij (M x N1), Lvq (N1), L_vec (1 x N1)
# Output dimension: M x N1
def g1_matrix(Q, d_edge, l_edge, sigma, D):
    Q_ji = np.transpose(Q)  # Edgewise operations, want rows as edges
    # Eq.(1) Aupetit 2005
    res = np.power(2 * np.pi * sigma**2, -(D - 1) / 2)
    res *= np.exp(-np.transpose(d_edge) ** 2 / (2 * sigma**2))
    res /= 2 * l_edge
    res *= erf(Q_ji / (sigma * np.sqrt(2))) - erf(
        (Q_ji - l_edge) / (sigma * np.sqrt(2))
    )
    # Return matrix with data as rows, edges as columns
    return np.transpose(res)


# Component probabilities
# Apply Bayes theorem to go from  P(component|data) to P(data|component)
# (The P's used in Aupetit eq. (4), denominator is eq. (2))
def p_matrix(pi, g0, g1):
    p0 = g0 * np.transpose(pi[0])
    p1 = g1 * np.transpose(pi[1])
    pxj = np.reshape(np.sum(p0, axis=1) + np.sum(p1, axis=1), (-1, 1))
    p0 /= pxj
    p1 /= pxj
    return (p0, p1)


