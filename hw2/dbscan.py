#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 20:12:10 2017
dbscan
@author: luminous
"""
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import matplotlib.pyplot as plt

"""implore data"""
track_matrix = []
track_matrix_file = open('track_matrix.txt', 'r')
for line in track_matrix_file:
    track_matrix.append([float(line)])
track_matrix_file.close()

"""DBSCAN"""
k = []
sil = []
max_k = 2
max_sil = 0.0
max_eps = 0.5
res_lable = []

eps = 0.01
max_dis = max(track_matrix)[0] - min(track_matrix)[0]
while (eps < max_dis):
    db = DBSCAN(eps=eps, min_samples=1).fit(track_matrix)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    if(n_clusters <= 1):
        break
    silhouette_avg = metrics.silhouette_score(track_matrix, db.labels_)
    eps += 0.01
    
    """print"""
    print "For eps =", eps, ": n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg
    k.append(n_clusters)
    sil.append(silhouette_avg)
    
    """result"""
    if silhouette_avg > max_sil and n_clusters > 3:
        max_sil = silhouette_avg
        max_k = n_clusters
        max_eps = eps
        res_label = db.labels_

plt.plot(k, sil)
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.show

res_file = open("dbscan_res.txt", "w")
res_file.write(str(max_k) + "\n")
for label in res_label:
    res_file.write(str(label) + " ")
res_file.close()