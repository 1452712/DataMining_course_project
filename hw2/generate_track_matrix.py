#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 15:34:27 2017
Generate track-matrix
@author: luminous
"""
import numpy as np
import pandas as pd
import time
from lshash import LSHash
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.decomposition import PCA 

"""pre-treat data"""
utm_data = pd.read_csv("Traj_1000_SH_UTM")
utm_data = pd.DataFrame(utm_data)
utm_data = dict(utm_data)
utm_track = {}
utm_grid = np.zeros([(362800 - 346000) / 2, (3463800 - 3448600) / 2], np.int8)
count = 0
current_tid = 0
for item in utm_data["Tid"]:
    if item != current_tid :
        utm_track[item] = []
        current_tid = item
    """transform time"""
    unix_time = time.mktime(time.strptime(utm_data["Time"][count],"%Y-%m-%d %H:%M:%S"))
    """grid"""
    x = int((utm_data["X"][count] - 346000) / 20)
    y = int((utm_data["Y"][count] - 3448600) / 20)
    utm_grid[x][y] += 1
    utm_track[item].append([unix_time, x, y])
    count += 1

"""identifier grid"""
count = 1
for x in range(len(utm_grid)):
    for y in range(len(utm_grid[x])):
        if utm_grid[x][y] != 0:
            utm_grid[x][y] = count
            count += 1

"""0-1 matrix"""
input_dim = count - 1
track_matrix = np.zeros([1000, input_dim], np.int8)
for track in utm_track:
    for item in utm_track[track]:
        track_matrix[track -1][utm_grid[item[1]][item[2]]] = 1

"""pca"""
pca=PCA(n_components=1)
track_matrix_pca=pca.fit_transform(track_matrix)

track_file = open("track_matrix.txt", "w")
for item in track_matrix_pca:
    print item[0]
    track_file.write(str(item[0]) + "\n")
track_file.close()


