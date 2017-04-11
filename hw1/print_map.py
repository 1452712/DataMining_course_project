#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 01:42:17 2017
print the map
@author: luminous
"""
import pandas as pd
import numpy as np
import time
from operator import itemgetter
from mpl_toolkits import Basemap  
import matplotlib.pyplot as plt  

gps_data = pd.read_csv("Traj_1000_SH_GPS")
gps_data = pd.DataFrame(gps_data)
gps_data = dict(gps_data)
gps_track = {}
count = 0
current_tid = 0
for item in gps_data["Tid"]:
    if item != current_tid :
        gps_track[item] = []
        current_tid = item
    """transform time"""
    unix_time = time.mktime(time.strptime(gps_data["Time"][count],"%Y-%m-%d %H:%M:%S"))
    gps_track[item].append([unix_time, gps_data["Lon"][count], gps_data["Lat"][count]])
    count += 1
    
m = Basemap(llcrnrlon=-100.,llcrnrlat=0.,urcrnrlon=-20.,urcrnrlat=57.,
            projection='lcc',lat_1=20.,lat_2=40.,lon_0=-60.,
            resolution ='l',area_thresh=1000.)
for item in gps_track:   
    """sort based on time"""
    sorted(gps_track[item], key = itemgetter(0))
    """sorted(item, cmp = lambda x,y: cmp(x[0],y[0]))"""
    for track in gps_track[item]:
        m.plot(track[1],track[2],color='k')
# draw coastlines, meridians and parallels.
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='#cc9966',lake_color='#99ffff')
m.drawparallels(np.arange(10,70,20),labels=[1,1,0,0])
m.drawmeridians(np.arange(-100,0,20),labels=[0,0,0,1])
plt.title('Track')
plt.show()

