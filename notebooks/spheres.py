import numpy as np
##
## This function is modified from https://github.com/BorgwardtLab/topological-autoencoders
## distributed under a BSD license: 
## https://github.com/BorgwardtLab/topological-autoencoders/blob/master/LICENSE
## Copyright (c) 2020, Michael Moor, Max Horn, Bastian Rieck
## 
def dsphere(n=100, d=2, r=1):

    data = np.random.randn(n, d+1)
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None]) 

    return data

def create_sphere_dataset(n_samples=500, d=3, n_spheres=11, r=5, plot=False, seed=42):
    np.random.seed(seed)

    variance=10/np.sqrt(d)
    shift_matrix = np.random.normal(0,variance,[n_spheres, d+1])

    spheres = [] 
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i,:])
        n_datapoints += n_samples

    #Additional big surrounding sphere:
    n_samples_big = 10*n_samples #int(n_samples/2)
    big = dsphere(n=n_samples_big, d=d, r=r*5)
    spheres.append(big)
    n_datapoints += n_samples_big

    #Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints) 
    label_index=0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples
    
    return dataset, labels 

