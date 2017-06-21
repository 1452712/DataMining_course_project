'''
Created on Jun 5, 2017

@author: dam
'''
import numpy as np
import pandas as pd
import math
from helloPython.helloPython import fileOut

XCellDimension = 3

def readGongcanList(filename):
    data = pd.read_csv(filename)
    outputLength = len(data)
    result = []
    X = []
    y = []
    result.append(X)
    result.append(y)
    i = 0
    while(i<outputLength):
        X.append([data.at[i, "LAC"],data.at[i, "CI"]])
        y.append([data.at[i, "Longitude"],data.at[i, "Latitude"]])
        i+=1
    return result

def readOriginList(filename, tosort = True):
    data = pd.read_csv(filename)
    outputLength = len(data)
    result = []
    i = 0
    while(i<outputLength):
        row = []
        row.append(data.at[i,"RNCID_1"]) # 0
        row.append(data.at[i,"CellID_1"]) # 1
        row.append(data.at[i,"RSCP_1"] - data.at[i,"EcNo_1"]) # 2
        row.append(data.at[i,"Longitude"]) # 3
        row.append(data.at[i,"Latitude"]) # 4
        result.append(row)
        i+=1
    if(tosort):
        result.sort(key=lambda r:(r[0],r[1]))
    return result

def readNewList(originname, gongcan):
    if(isinstance(gongcan,str)):
        gongcan = readGongcanList(gongcan)
    elif(not isinstance(gongcan,list)):
        return None
    gongcanX = gongcan[0]
    gongcany = gongcan[1]
    origin = readOriginList(originname)
    group = [[-9999,-9999]]
    result = []
    for row in origin:
        if(group[0][0]!=row[0] or group[0][1]!=row[1]):
            result.append(group)
            group = []
            X=[]
            y=[]
            group.append([row[0],row[1]])
            group.append(X)
            group.append(y)
            indexGongcan = gongcanX.index([row[0],row[1]])
            group[0].append(gongcany[indexGongcan][0])
            group[0].append(gongcany[indexGongcan][1])
        X.append([row[2]])
        y.append([row[3]-group[0][2],row[4]-group[0][3]])
    result.append(group)
    del result[0]
    return result

#list: [[[LAC,CI,Lon,Lat],[[x]],[[yLon,yLat]]]]
def getTrTeListPair(gongcan, trName, teName, tofiltrate = False):
    if(isinstance(gongcan,str)):
        gongcan = readGongcanList(gongcan)
    elif(not isinstance(gongcan,list)):
        return None
    trList = readNewList(trName, gongcan)
    teList = readNewList(teName, gongcan)
    if(tofiltrate):
        i = 0
        while(i != min(len(trList),len(teList))):
            cmpResult = cmpTListRow(trList[i], teList[i])
            if(cmpResult>0):
                del teList[i]
            elif(cmpResult<0):
                del trList[i]
            else:
                i+=1
        while(i != len(trList)):
            del trList[i]
        while(i != len(teList)):
            del teList[i]
    return trList, teList

def cmpTListRow(x, y):
    result = x[0][0] - y[0][0]
    if result == 0:
        result = x[0][1] - y[0][1]
    return result


def getDistance(x, y, length=None):
    distance = 0
    if(length==None):
        length = len(x)
    for i in range(length):
        distance += (x[i]-y[i])*(x[i]-y[i])
    return math.sqrt(distance)

def getDistanceList(xList, yList, length=None):
    listLength = min(len(xList),len(yList))
    if(listLength>0 and length==None):
        length = len(xList[0])
    result = []
    for i in range(listLength):
        result.append(getDistance(xList[i], yList[i], length))
    return result

def getAverageList(colList):
    colNum = len(colList)
    length = len(colList[0])
    return [np.average([colList[i][j] for i in range(colNum)]) for j in range(length)]
    
def getMedian(myList ,flagSorted = False):
    length = len(myList)
    if(length == 0):
        return None
    tempList = myList
    if(not flagSorted):
        sortedList = [i for i in myList]
        sortedList.sort()
        tempList = sortedList
    if(length%2 == 0):
        return (tempList[length/2-1] + tempList[length/2])/2
    else:
        return tempList[(length-1)/2]
  
'''
# gongcan = readGongcanList("final_2g_gongcan.csv")
# result = readNewList("final_2g_tr.csv",gongcan)
# resultTe = readNewList("final_2g_te.csv")
result, resultTe = getTrTeListPair("final_2g_gongcan.csv", "final_2g_tr.csv", "final_2g_te.csv", True)
for i in range(min(len(result),len(resultTe))):
    print result[i][0],len(result[i][1]),len(result[i][2]),resultTe[i][0],len(resultTe[i][1]),len(resultTe[i][2])
print len(result),len(resultTe)
# for row in result:
#     print row[0],len(row[1]),len(row[2])
# for i in range(len(result[0][1])):
#     print result[0][1][i],result[0][2][i]
'''
def main00():
    data = pd.read_csv("final_2g_gongcan.csv")
    file_out = open("result/my_gongcan_2g.csv", "w")
    file_out.write("id,")
    for c in data.columns:
        file_out.write(c+",")
    file_out.write("\r\n")
    result_tr, result_te = getTrTeListPair("final_2g_gongcan.csv", "final_2g_tr.csv", "final_2g_te.csv", True)
    for i in range(len(result_tr)):
        file_out.write(str(i) + ",")
        for j in range(len(data)):
            if(data.at[j, "LAC"] == result_tr[i][0][0] and data.at[j, "CI"] == result_tr[i][0][1]):
                break
        for c in data.columns:
            file_out.write(str(data.at[j,c])+",")
        file_out.write("\r\n")
    return
    file_out.close()
main00()