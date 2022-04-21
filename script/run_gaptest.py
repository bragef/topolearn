
#%%

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as pl
from numpy.random import random_sample
from topolearn.util.cluster import KMeansGap


X, y = make_blobs(n_samples=1000, centers=7, n_features=2)

pl.scatter(X[:, 0], X[:, 1], c=y, s=2)

learner = KMeansGap(gap_iter = 5)
learner.fit(X)

labels = learner.fit(X).predict(X)

pl.figure()
pl.plot(learner.gap)







# %%
