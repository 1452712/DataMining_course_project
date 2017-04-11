#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 21:41:01 2017
knn
TODO: multi-thread
@author: luminous
"""
import numpy as np
import pandas as pd
import time
from lshash import LSHash
from sklearn.neighbors import KNeighborsClassifier as knn

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
"""lsh"""
hash_size_list = [10, 11, 12, 13, 14, 15]
query_track_list = [15, 250, 480, 690, 900]
output = open("result.txt", "w")
for hash_size in hash_size_list:
    output.write("hash_size: %d\n" % hash_size)
    lsh = LSHash(hash_size, input_dim)
    for track_num in range(1000):
        lsh.index(track_matrix[track_num])
    for query_track in query_track_list:
        output.write("query_track: %d\n" % query_track)
        output.writelines(str(lsh.query(track_matrix[query_track - 1])))
output.close()

"""knn"""
n_neighbors = 1
weights = 'uniform'
"""optional: distance"""
algo = 'auto'
"""optional: ball_tree/kd_tree/brute"""
neigh = knn(n_neighbors, weights, algo)
for index in range(1000):
    temp = np.zeros(1000)
    temp[index] = 1
    neigh.fit(track_matrix, temp)
for query_track in query_track_list:
    print neigh.kneighbors(track_matrix[query_track - 1])


    

