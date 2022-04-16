# %%

import pandas as pd
import numpy as np


pairwise_lod = pd.read_csv("../popgenetics/blueskate/hfc-logl-nocs.csv")
pairwise_diss_dist =  np.loadtxt(open("../popgenetics/blueskate/diss-dist.csv"), delimiter=" ", skiprows=1)
pairwise_features = pd.read_csv("../popgenetics/blueskate/covars.csv")
#pairwise_features[ pairwise_features["lat_dec"]

# Define max_logl - logl as distance, this ensures all distances are positive.
# max_logl should be set high enough that it only include same indidivual. 
# (might not always be possible, since there may be overlap in the logls if there are not enough markers)
self_distance = 100 
pairwise_lod["dist"] = np.maximum(0,  self_distance-pairwise_lod["logl_ratio"] )

# Now we need a distance matrix from the pairs. There is probabaly some more clever way to do this
dist_X = np.zeros((pairwise_features.shape[0], pairwise_features.shape[0]))
sid_idx = { sid:idx for idx, sid in zip(pairwise_features.index, pairwise_features["sample_ID"])  }

for index, row in pairwise_lod.iterrows():
    idx_1, idx_2 = sid_idx[row["D1_indiv"]], sid_idx[row["D2_indiv"]] 
    dist_X[idx_1, idx_2] = dist_X[idx_2, idx_1] = row["dist"]


# %%
# For the visualisations: We add some jitter to the geographic coordinates
# (sigma=0.2 ~ 20km)
n = pairwise_features["lat_dec"].shape[0]
lat =  np.array(pairwise_features["lat_dec"]) + np.random.normal(0,0.2, n)
long =  np.array(pairwise_features["long_dec"]) + np.random.normal(0,0.2, n) 
X_geo = np.array([long, lat]).T



# %%
import matplotlib.pyplot as pl
import numpy as np
import networkx as nx
from topolearn import homology as ph
from topolearn import util
import importlib
importlib.reload(ph)



learner = ph.RipsComplex(max_dim = 2, max_radius= 2900, num_steps=200)

dist_X = pairwise_diss_dist 

simplices = learner.fit(dist_X)
graph = simplices.graph( X_geo)

nx.draw(graph, pos=X_geo, node_color="r", edge_color="k", alpha=0.01, node_size=2)

#util.plot_graph_with_data(graph, X_geo)

#nx.draw(graph, pos=X_geo, node_color="r", edge_color="k", alpha=0.05, node_size=2)





# %%
