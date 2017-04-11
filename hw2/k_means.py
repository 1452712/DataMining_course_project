#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:30:04 2017
k-means
@author: luminous
"""
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

"""implore data"""
track_matrix = []
track_matrix_file = open('track_matrix.txt', 'r')
for line in track_matrix_file:
    track_matrix.append([float(line)])
track_matrix_file.close()

"""k-means"""
init_k = int(1000 ** 0.5)
k = []
sil = []
max_k = 2
max_sil = 0.0
res_lable = []

for n_clusters in range(5, init_k + 1):
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(track_matrix)
    silhouette_avg = silhouette_score(track_matrix, cluster_labels)
    
    """print"""
    print "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg
    k.append(n_clusters)
    sil.append(silhouette_avg)
    
    """result"""
    if silhouette_avg > max_sil and n_clusters > 5:
        max_sil = silhouette_avg
        max_k = n_clusters
        res_label = cluster_labels        

plt.plot(k, sil)
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.show

res_file = open("k_means_res.txt", "w")
res_file.write(str(max_k) + "\n")
for label in res_label:
    res_file.write(str(label) + " ")
res_file.close()