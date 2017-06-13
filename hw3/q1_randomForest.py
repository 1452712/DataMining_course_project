#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  26 16:11:13 2017
Q1: Random Forest
@author: luminous
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# import data
def importData(inFile):      
    data = pd.read_csv(inFile)
    out = {}
    out["RSSI"] = []
    out["Grid"] = []

    minLon = 121.47738 # min(data["Longitude"])
    maxLon = 121.5025075 # max(data["Longitude"])\
    lonCount = math.ceil((maxLon - minLon) / 0.0001)

    minLat = 31.20891667 # min(data["Latitude"])
    maxLat = 31.219175 # max(data["Latitude"])
    latCount = math.ceil((maxLat - minLat) / 0.0001)

    for i in range(len(data)):
        # Strength of Signal
        # RSSI = RSCP – EcNo
        out["RSSI"].append([data["RSCP_1"][i] - data["EcNo_1"][i]])
        # GPS Grid ID
        x = int((data["Longitude"][i] - minLon) / 0.0001)
        y = int((data["Latitude"][i] - minLat) / 0.0001)
        out["Grid"].append(int(x + y * lonCount))
 
    return out

# calculate accuracy
def getAccuracy(res, target):
    if len(res) != len(target):
        return 0
    num = len(res)
    count = 0
    for i in range(num):
        if int(res[i]) == target[i]:
            count = count + 1
    return float(count) / float(num) * 100

"""
"main" function
"""
# LTE
#train_data = importData("new4gtrain.csv")
#test_data = importData("new4gtest.csv")
#estimators = 7
# GSM
train_data = importData("new2gtrain.csv")
test_data = importData("new2gtest.csv")
cla_estimators = 10
reg_estimators = 10

# Random Forest Classifier
acc = []
for i in range(10):
    classifier = RandomForestClassifier(n_estimators=cla_estimators)
    classifier.fit(train_data["RSSI"], train_data["Grid"])
    cla_res = classifier.predict(test_data["RSSI"])
    cla_accuracy = getAccuracy(cla_res, test_data["Grid"])
    acc.append(cla_accuracy)
acc.sort()
plt.plot(acc, range(1, 11))
plt.xlabel("No.")
plt.ylabel("Accuracy")
plt.show
print "Med-accuracy of Random Forest Classifier is %d" % acc[6]

# Random Forest Regressor
reg_acc = []
for i in range(10):
    regressor = RandomForestRegressor()
    regressor.fit(train_data["RSSI"], train_data["Grid"])
    reg_res = regressor.predict(test_data["RSSI"])
    reg_accuracy = getAccuracy(reg_res, test_data["Grid"])
    reg_acc.append(reg_accuracy)
reg_acc.sort()
plt.plot(reg_acc, range(1, 11))
plt.xlabel("No.")
plt.ylabel("Accuracy")
plt.show
print "Med-accuracy of Random Forest Regressor is %d" % reg_acc[6]

