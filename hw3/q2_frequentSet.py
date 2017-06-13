#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  27 00:00:56 2017
Frequent Set Mining
@author: luminous
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# import data
def importData(inFile):      
    data = pd.read_csv(inFile)
    out = {}
    out["ID"] = []
    out["Grid"] = []

    minLon = 121.47738 # min(data["Longitude"])
    maxLon = 121.5025075 # max(data["Longitude"])\
    lonCount = math.ceil((maxLon - minLon) / 0.0001)

    minLat = 31.20891667 # min(data["Latitude"])
    maxLat = 31.219175 # max(data["Latitude"])
    latCount = math.ceil((maxLat - minLat) / 0.0001)

    for i in range(len(data)):
        # ID of Tel.
        out["ID"].append([data["IMSI"][i]])
        # GPS Grid ID
        x = int((data["Longitude"][i] - minLon) / 0.0001)
        y = int((data["Latitude"][i] - minLat) / 0.0001)
        out["Grid"].append(int(x + y * lonCount))
 
    return out

