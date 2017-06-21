'''
Created on Jun 15, 2017
Content:
1. Pretreatment of Data
2. Drawing
@author: dam,
@update: luminous
'''

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def drawError(single_error,double_error,flag_show=False,flag_save=False,title=None,filename=None):
    if(flag_show or flag_save):
        plt.figure()
        s = 20
        a = 0.4
        single_route_id = range(len(single_error))
        double_route_id = range(len(double_error))
        plt.scatter(single_error, single_route_id,
                    c="red", s=s, marker="o", alpha=a, label="Single HMM")
        plt.scatter(double_error, double_route_id,
                    c="navy", s=s, marker="s", alpha=a, label="Double HMM")
        xmin = min(min(single_error),min(double_error))
        xmax = max(max(single_error),max(double_error))
        ymin = min(min(single_route_id), min(double_route_id))
        ymax = max(max(single_route_id), max(double_route_id))
        xlength = xmax - xmin
        ylength = ymax - ymin
        figure_adjust = 0.2
        plt.xlim([xmin-figure_adjust*xlength,xmax+figure_adjust*xlength])
        plt.ylim([ymin-figure_adjust*ylength,ymax+figure_adjust*ylength])
        plt.xlabel("Error")
        plt.ylabel("Route")
        plt.title(title)
        plt.legend()
        if(flag_save):
            plt.savefig(filename)
        if(flag_show):
            plt.show()
        plt.close()
    return

def drawPreAct(y_train,y_test,flag_show=False,flag_save=False,title=None,filename=None):
    if(flag_show or flag_save):
        plt.figure()
        s = 20
        a = 0.4
        y_train_x = [row[0] for row in y_train]
        y_train_y = [row[1] for row in y_train]
        y_test_x = [row[0] for row in y_test]
        y_test_y = [row[1] for row in y_test]
        plt.scatter(y_test_x, y_test_y,
                    c="navy", s=20, marker="s", alpha=0.4, label="Reality")
        plt.scatter(y_train_x, y_train_y,
                    c="red", s=s, marker="o", alpha=a, label="Prediction")
        xmin = min(min(y_train_x),min(y_test_x))
        xmax = max(max(y_train_x),max(y_test_x))
        ymin = min(min(y_train_y),min(y_test_y))
        ymax = max(max(y_train_y),max(y_test_y))
        xlength = xmax - xmin
        ylength = ymax - ymin
        figure_adjust = 0.2
        plt.xlim([xmin-figure_adjust*xlength,xmax+figure_adjust*xlength])
        plt.ylim([ymin-figure_adjust*ylength,ymax+figure_adjust*ylength])
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(title)
        plt.legend()
        if(flag_save):
            plt.savefig(filename)
        if(flag_show):
            plt.show()
        plt.close()

def processOriginDataForP2(inputname,outputname,cell_length=0.0005,method="ChangeBases",path_distance=0.0005):
    if(method == "ChangeBases"):
        return processOriginDataForP2ChangeBases(inputname, outputname, cell_length)
    elif(method == "Distance"):
        return processOriginDataForP2Distance(inputname, outputname, cell_length, path_distance)
    
def processOriginDataForP2ChangeBases(inputname,outputname,cell_length=0.0005):
    data = pd.read_csv(inputname)
    longitudes = [data.at[i,"Longitude"] for i in range(len(data))]
    latitudes = [data.at[i,"Latitude"] for i in range(len(data))]
    x_min = min(longitudes)
    x_max = max(longitudes)
    y_min = min(latitudes)
    y_max = max(latitudes)
    fileOut = open(outputname, "w")
    fileOut.write("PathID,RNCID_1,CellID_1,EcNo_1,RSCP_1,RTT_1,UE_Rx_Tx_1,RSSI_1,RNCID_2,CellID_2,EcNo_2,RSCP_2,RTT_2,UE_Rx_Tx_2,RSSI_2,RNCID_3,CellID_3,EcNo_3,RSCP_3,RTT_3,UE_Rx_Tx_3,RSSI_3,RNCID_4,CellID_4,EcNo_4,RSCP_4,RTT_4,UE_Rx_Tx_4,RSSI_4,RNCID_5,CellID_5,EcNo_5,RSCP_5,RTT_5,UE_Rx_Tx_5,RSSI_5,RNCID_6,CellID_6,EcNo_6,RSCP_6,RTT_6,UE_Rx_Tx_6,RSSI_6,Longitude,Latitude,Cell_ID_x,Cell_ID_y,Cell_center_x,Cell_center_y,Grid_ID,Grid_center_x,Grid_center_y,\r\n")
    pathID = -1
    baseCodes = [-999 for i in range(12)]#current base RNCID and CellID
    for i in range(len(data)):
        flagChangePath = False
        for j in range(1,7):
            if(data.at[i,"RNCID_"+str(j)] != baseCodes[j*2-2] or data.at[i,"CellID_"+str(j)] != baseCodes[j*2-1]):
                flagChangePath = True
                break
        if(flagChangePath):
            pathID += 1
            for j in range(1,7):
                baseCodes[j*2-2] = data.at[i,"RNCID_"+str(j)]
                baseCodes[j*2-1] = data.at[i,"CellID_"+str(j)]
        fileOut.write(str(pathID)+",")
        for j in range(1,7):
            fileOut.write(str(data.at[i,"RNCID_"+str(j)])+",")
            fileOut.write(str(data.at[i,"CellID_"+str(j)])+",")
            fileOut.write(str(data.at[i,"EcNo_"+str(j)])+",")
            fileOut.write(str(data.at[i,"RSCP_"+str(j)])+",")
            fileOut.write(str(data.at[i,"RTT_"+str(j)])+",")
            fileOut.write(str(data.at[i,"UE_Rx_Tx_"+str(j)])+",")
            fileOut.write(str(data.at[i,"RSCP_"+str(j)]-data.at[i,"EcNo_"+str(j)])+",")
        '''Longitude,Latitude,Cell_ID_x,Cell_ID_y,Cell_center_x,Cell_center_y,Grid_ID,Grid_center_x,Grid_center_y'''
        longitude = data.at[i,"Longitude"]
        latitude = data.at[i,"Latitude"]
        fileOut.write(str(longitude)+",")
        fileOut.write(str(latitude)+",")
        cell_ID_x = int((longitude-x_min)/cell_length)
        cell_ID_y = int((latitude-y_min)/cell_length)
        cell_center_x = x_min + (float(cell_ID_x)+0.5)*cell_length
        cell_center_y = y_min + (float(cell_ID_y)+0.5)*cell_length
        fileOut.write(str(cell_ID_x)+",")
        fileOut.write(str(cell_ID_y)+",")
        fileOut.write(str(cell_center_x)+",")
        fileOut.write(str(cell_center_y)+",")        
        fileOut.write(str(data.at[i,"Grid_ID"])+",")
        fileOut.write(str(data.at[i,"Grid_center_x"])+",")
        fileOut.write(str(data.at[i,"Grid_center_y"])+",")
        fileOut.write("\r\n")
    fileOut.close()
    return [x_min, x_max], [y_min, y_max]

def processOriginDataForP2Distance(inputname,outputname,cell_length=0.0005,path_distance=0.0005):
    data = pd.read_csv(inputname)
    longitudes = [data.at[i,"Longitude"] for i in range(len(data))]
    latitudes = [data.at[i,"Latitude"] for i in range(len(data))]
    x_min = min(longitudes)
    x_max = max(longitudes)
    y_min = min(latitudes)
    y_max = max(latitudes)
    fileOut = open(outputname, "w")
    fileOut.write("PathID,RNCID_1,CellID_1,EcNo_1,RSCP_1,RTT_1,UE_Rx_Tx_1,RSSI_1,RNCID_2,CellID_2,EcNo_2,RSCP_2,RTT_2,UE_Rx_Tx_2,RSSI_2,RNCID_3,CellID_3,EcNo_3,RSCP_3,RTT_3,UE_Rx_Tx_3,RSSI_3,RNCID_4,CellID_4,EcNo_4,RSCP_4,RTT_4,UE_Rx_Tx_4,RSSI_4,RNCID_5,CellID_5,EcNo_5,RSCP_5,RTT_5,UE_Rx_Tx_5,RSSI_5,RNCID_6,CellID_6,EcNo_6,RSCP_6,RTT_6,UE_Rx_Tx_6,RSSI_6,Longitude,Latitude,Cell_ID_x,Cell_ID_y,Cell_center_x,Cell_center_y,Grid_ID,Grid_center_x,Grid_center_y,\r\n")
    pathID = -1
    longitude = 361
    latitude = 361
    for i in range(len(data)):
        if(getDistance([longitude,latitude],[data.at[i,"Longitude"],data.at[i,"Latitude"]])>path_distance):
            pathID += 1
        longitude = data.at[i,"Longitude"]
        latitude = data.at[i,"Latitude"]
        fileOut.write(str(pathID)+",")
        for j in range(1,7):
            fileOut.write(str(data.at[i,"RNCID_"+str(j)])+",")
            fileOut.write(str(data.at[i,"CellID_"+str(j)])+",")
            fileOut.write(str(data.at[i,"EcNo_"+str(j)])+",")
            fileOut.write(str(data.at[i,"RSCP_"+str(j)])+",")
            fileOut.write(str(data.at[i,"RTT_"+str(j)])+",")
            fileOut.write(str(data.at[i,"UE_Rx_Tx_"+str(j)])+",")
            fileOut.write(str(data.at[i,"RSCP_"+str(j)]-data.at[i,"EcNo_"+str(j)])+",")
        '''Longitude,Latitude,Cell_ID_x,Cell_ID_y,Cell_center_x,Cell_center_y,Grid_ID,Grid_center_x,Grid_center_y'''
        fileOut.write(str(longitude)+",")
        fileOut.write(str(latitude)+",")
        cell_ID_x = int((longitude-x_min)/cell_length)
        cell_ID_y = int((latitude-y_min)/cell_length)
        cell_center_x = x_min + (float(cell_ID_x)+0.5)*cell_length
        cell_center_y = y_min + (float(cell_ID_y)+0.5)*cell_length
        fileOut.write(str(cell_ID_x)+",")
        fileOut.write(str(cell_ID_y)+",")
        fileOut.write(str(cell_center_x)+",")
        fileOut.write(str(cell_center_y)+",")        
        fileOut.write(str(data.at[i,"Grid_ID"])+",")
        fileOut.write(str(data.at[i,"Grid_center_x"])+",")
        fileOut.write(str(data.at[i,"Grid_center_y"])+",")
        fileOut.write("\r\n")
    fileOut.close()
    return [x_min, x_max], [y_min, y_max]

def getDistance(x,y):
    s = 0
    for i in range(len(x)):
        s += (x[i]-y[i])*(x[i]-y[i])
    return math.sqrt(s)

def drawPath(path_x,path_y=None,flag_show=False,flag_save=False,title=None,filename=None):
    if(flag_show or flag_save):
        plt.figure()
        s = 20
        a = 0.4
        if(path_y == None):
            path_main = path_x
            path_x = [row[0] for row in path_main]
            path_y = [row[1] for row in path_main]
        plt.scatter(path_x, path_y,
                    c="red", s=s, marker="o", alpha=a, label="Path")
        xmin = min(path_x)
        xmax = max(path_x)
        ymin = min(path_y)
        ymax = max(path_y)
        xlength = xmax - xmin
        ylength = ymax - ymin
        figure_adjust = 0.2
        plt.xlim([xmin-figure_adjust*xlength,xmax+figure_adjust*xlength])
        plt.ylim([ymin-figure_adjust*ylength,ymax+figure_adjust*ylength])
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(title)
        plt.legend()
        if(flag_show):
            plt.show()
        if(flag_save):
            plt.savefig(filename)
        plt.close()
    return

def drawPathForCsv(inputname,drawRange=None,flag_show=False,flag_save=False,title=None,filename=None):
    data = pd.read_csv(inputname)
    paths = []
    pathID = -1
    for i in range(len(data)):
        if(data.at[i,"PathID"]!=pathID):
            pathID = data.at[i,"PathID"]
            paths.append([])
        paths[pathID].append([data.at[i,"Longitude"],data.at[i,"Latitude"]])
    if(drawRange==None):
        drawRange = range(len(paths))
    for i in drawRange:
        drawPath(paths[i], None, flag_show, flag_save, title+str(i), filename+str(i))
#print processOriginDataForP2("final_2g_tr.csv","result/result_p2_0/my_final_2g_tr.csv")
#print processOriginDataForP2("final_4g_tr.csv","result/result_p2_0/my_final_4g_tr.csv")
#print processOriginDataForP2("final_2g_tr.csv","result/result_p2_0/my_final_2g_tr_2.csv",method="Distance")
#print processOriginDataForP2("final_4g_tr.csv","result/result_p2_0/my_final_4g_tr_2.csv",method="Distance")
#drawPathForCsv("result/result_p2_0/my_final_2g_tr.csv", [0,1,2,3,4], False, True, "2g img ", "result/result_p2_0/input_2g_path_ChangeBases_")