#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 19:49:40 2017
print Baidu Map
@author: luminous
"""
import pandas as pd
"""implore data"""
res_file = open("k_means_res.txt", "r")
#res_file = open("dbscan_res.txt", "r")
k = int(res_file.readline())
str_label = res_file.readline()
res_file.close()
label = str_label.split(" ")
"""remove the last space"""
if label[len(label) - 1] == "":
    label.pop(len(label) - 1)

"""implore original gps data"""
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
    gps_track[item].append([gps_data["Lon"][count], gps_data["Lat"][count]])
    count += 1

"""Generate HTML"""
_author_ = "Luminous"
_time_  = "Apr 8 2017"
import webbrowser

GEN_HTML = "k_means.html"
#GEN_HTML = "dbscan.html"

f = open(GEN_HTML,'w')
message = """
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="initial-scale=1.0, user-scalable=no" />
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  <style type="text/css">
    body, html{width: 100%; height: 100%; margin:0; font-family:"微软雅黑";}
    #my_map{height:500px;width:100%;}
  </style>
  <script src="http://d3js.org/d3.v3.min.js" charset="utf-8"></script> 
  <script type="text/javascript" src="http://api.map.baidu.com/api?v=2.0&ak=kShcg14DfaxTHwCmVl0wKGRCDwm5DpGd"></script>
  <title>Result of Clustering</title>
</head>
<body>
  <div style="height:100%" id="my_map"></div>
</body>
</html>

<script type="text/javascript">
  var map = new BMap.Map("my_map");
  var point = new BMap.Point(121.512615, 31.212465);
  map.centerAndZoom(point, 12);
"""
"""draw lines"""
count = 0
for item in gps_track:
    message += "var line = [];"
    for point in gps_track[item]:
        message += "line.push(new BMap.Point(%f, %f));"%(point[0],point[1])
    if label[count] == '0':
        color = "red"
    elif label[count] == '1':
        color = "blue"
    elif label[count] == '2':
        color = "yellow"
    elif label[count] == '3':
        color = "green"
    elif label[count] == '4':
        color = "black"
    elif label[count] == '5':
        color = "grey"
    elif label[count] == '6':
        color = "white"
    elif label[count] == '7':
        """ current max """
        color = "purple"
    message += """
    var polyline = new BMap.Polyline(line, {strokeColor: "%s", strokeWeight:2, strokeOpacity:0.5});
    map.addOverlay(polyline);
    """%(color)
    count += 1

message += """
  map.enableScrollWheelZoom(true);
</script>
"""

f.write(message)
f.close()

webbrowser.open(GEN_HTML,new = 1)
