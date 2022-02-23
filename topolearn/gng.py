
import numpy as np
import networkx as nx
import random

## Growing Neural Gas algorithm. 



class GNG():

    def __init__(
        self,
        alpha = 0.01,            # Learning rate
        beta = 0.75,       # Error decay 
        gamma = 0.995,     # Error decay
        max_age = 10,           # Age threshold
        k = 10,                 # Num. epochs 
        max_iter = 10,
        max_nodes = 200, 
        m = 2,
        debug = 1
    ):        
        self.alpha = alpha
        self.k = k
        self.max_age = max_age
        self.max_iter = max_iter
        self.max_nodes = max_nodes
        self.beta = beta
        self.gamma = gamma
        self.debug = debug
        self.m = 300
        self.nodecounter = 0
        self.GG = None

    def fit(self, X, y = None):

        # Growing Neural Gas algorithm. 
        # Implemented as described in Fritzke 1995: A Growing Neural Gas Network Learns Topologies
        # D = X.shape[1] # Data dimension 
        GG = nx.Graph()

        # Initialise with two random, connected,  nodes 
        X_init = X[random.sample(range(0, len(X)), 2)]
        self.nodecounter += 1
        GG.add_node(self.nodecounter, w = X_init[0], err = 0)
        self.nodecounter += 1
        GG.add_node(self.nodecounter, w = X_init[1], err = 0)            
        GG.add_edge(1, 2, age=0)
        
        for epoch in range(0, self.max_iter):
            steps = 0
            for i in range(0, len(X)):
                point=X[i]
                # Get the euclidian distances from point to each node
                distances = [(np.linalg.norm(d['w'] - point), n) for n,d in GG.nodes(data=True)]  
                # Tuples with (distance, nodeid) for the two closest nodes
                closest = sorted(distances)[0:2]    
                node_1, node_2 = closest[0][1], closest[1][1]
                # Update age of all edges                
                for n1, n2, attribs in GG.edges(node_1, data=True):
                    GG.add_edge(n1, n2, age = attribs["age"] + 1)
                # Update error 
                GG.nodes[node_1]["err"] +=  closest[0][0]
                # Nudge the two nearest nodes closer to this point
                # Note symmetric learning rate here, original algorithm uses epsilon_n, epsilon_d
                GG.nodes[node_1]["w"] +=  self.alpha * (point - GG.nodes[node_1]["w"]) 
                GG.nodes[node_2]["w"] +=  self.alpha * (point - GG.nodes[node_2]["w"]) 
                # Age 0 for edge between closest nodes
                GG.add_edge(node_1, node_2, age=0)
                # Remove edges past max_age
                for n1, n2, attribs in list(GG.edges(node_1, data=True)):
                    if(attribs["age"] > self.max_age):
                        GG.remove_edge(n1, n2)
                # Remove unconnected nodes
                GG.remove_nodes_from(nx.isolates(GG)) 

                steps += 1
            
                if steps % self.m == 0  and GG.number_of_nodes() < self.max_nodes:
                    if self.debug>0:
                        print("Epoch ", epoch)
                    # (max of tuples is the same as max of first element)
                    err_max, node_max_err  = max( [ (d["err"], n ) for n,d in GG.nodes(data=True) ] )
                    node_max_err_neigh  = max( [ (GG.nodes[n]["err"], n ) for n in GG.neighbors(node_max_err)])[1]
                    self.nodecounter += 1 
                    # Add node at midpoint between node with highest error and its neighbour with highest error
                    w_new  = ( GG.nodes[node_max_err]["w"] +  GG.nodes[node_max_err_neigh]["w"]  )/2
                    GG.add_node(self.nodecounter, w=w_new, err=err_max * self.beta)   
                    # Replace the direct edge between the nodes with an indirect edge
                    GG.remove_edge(node_max_err, node_max_err_neigh)
                    GG.add_edge(self.nodecounter, node_max_err, age=0)
                    GG.add_edge(self.nodecounter, node_max_err_neigh, age=0)
                    # Shrink the accumulated error of the nodes by a factor beta 
                    GG.nodes[node_max_err]["err"] *= self.beta
                    GG.nodes[node_max_err_neigh]["err"] *= self.beta

                    print(f"Nodes: {GG.number_of_nodes()}/{self.nodecounter}")

                if  GG.number_of_nodes() >= self.max_nodes:

                    print(f"Reached maximum number of nodes.")
                    break

            for n1 in GG.nodes():
                GG.nodes[n1]["err"] *= self.gamma

            self.GG = GG

        return self.GG

        # Random start nodes




    



