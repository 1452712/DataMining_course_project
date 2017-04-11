#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:50:19 2017
GMM
@author: luminous
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

"""implore data"""
track_matrix = []
track_matrix_file = open('track_matrix.txt', 'r')
for line in track_matrix_file:
    track_matrix.append([float(line)])
track_matrix_file.close()

"""result of k-means"""
km_res_file = open("k_means_res.txt", "r")
km_k = int(km_res_file.readline())
km_str_label = km_res_file.readline()
km_res_file.close()
km_char_label = km_str_label.split(" ")
km_label = []
"""remove the last space"""
if km_char_label[len(km_char_label) - 1] == "":
    km_char_label.pop(len(km_char_label) - 1)
for label in km_char_label:
    km_label.append(int(label))
    
    
"""result of DBSCAN"""
db_res_file = open("dbscan_res.txt", "r")
db_k = int(db_res_file.readline())
db_str_label = db_res_file.readline()
db_res_file.close()
db_char_label = db_str_label.split(" ")
db_label = []
"""remove the last space"""
if db_char_label[len(db_char_label) - 1] == "":
    db_char_label.pop(len(db_char_label) - 1)
for label in db_char_label:
    db_label.append(int(label))


""" GMM v.s. k-means"""
X_train = np.array(track_matrix)
y_train = np.array(km_label)
n_classes = km_k

estimator = GaussianMixture(n_components=n_classes, random_state=10)
estimator.means_init = np.array([X_train[y_train == i].mean(axis=0) for i in range(n_classes)])
estimator.fit(X_train)
y_train_pred = estimator.predict(X_train)
train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
print 'GMM vs k-means accuracy: %.1f' % train_accuracy


""" GMM v.s. DBSCAN"""
X_train = np.array(track_matrix)
y_train = np.array(db_label)
n_classes = db_k

estimator = GaussianMixture(n_components=n_classes, random_state=10)
estimator.means_init = np.array([X_train[y_train == i].mean(axis=0) for i in range(n_classes)])
estimator.fit(X_train)
y_train_pred = estimator.predict(X_train)
train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
print 'GMM vs DBSCAN accuracy: %.1f' % train_accuracy

