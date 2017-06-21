'''
Created on Jun 5, 2017

@author: dam
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import gc

import hw4_p1_00

def getRandomForestRegressorTrained(X_train, y_train, max_depth = 30, random_state = 1):
    regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
    regr_rf.fit(X_train, y_train)
    return regr_rf

def myRandomForestRegressorFit(regr_rf,X_test,y_test,flag_show=False,flag_save=False,title=None,filename=None):
    y_rf = regr_rf.predict(X_test)
    score = regr_rf.score(X_test, y_test)
    if(flag_show or flag_save):
        plt.figure()
        s = 50
        a = 0.4
        y_test_x = [row[0] for row in y_test]
        y_test_y = [row[1] for row in y_test]
        y_rf_x = [row[0] for row in y_rf]
        y_rf_y = [row[1] for row in y_rf]
        plt.scatter(y_test_x, y_test_y,
                    c="navy", s=s, marker="s", alpha=a, label="Data")
        plt.scatter(y_rf_x, y_rf_y,
                    c="c", s=s, marker="^", alpha=a,
                    label="RFR score=%.2f" % score)
        xmin = min(min(y_test_x),min(y_rf_x))
        xmax = max(max(y_test_x),max(y_rf_x))
        ymin = min(min(y_test_y),min(y_rf_y))
        ymax = max(max(y_test_y),max(y_rf_y))
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
    return y_rf, score

def drawDistances(sortedList,flag_show=False,flag_save=False,title=None,filename=None,max_y=0.005):
    length = len(sortedList)
    denominitor = length - 1;
    x = [float(i)/denominitor for i in range(length)]
    if(flag_show or flag_save):
        plt.figure()
        label = "Median="+str(hw4_p1_00.getMedian(sortedList,True))
        plt.plot(x,sortedList,label=label,
                    color="blue",linewidth=2,marker="o",markersize=2)
        plt.xlim([0,1])
        plt.ylim([0,max_y])
        plt.xlabel("Percent")
        plt.ylabel("Distance")
        plt.title(title)
        plt.legend()
        if(flag_show):
            plt.show()
        if(flag_save):
            plt.savefig(filename)
        plt.close()
    return     

def drawTrTe(y_train,y_test,flag_show=False,flag_save=False,title=None,filename=None):
    if(flag_show or flag_save):
        plt.figure()
        s = 20
        a = 0.4
        y_train_x = [row[0] for row in y_train]
        y_train_y = [row[1] for row in y_train]
        y_test_x = [row[0] for row in y_test]
        y_test_y = [row[1] for row in y_test]
        plt.scatter(y_train_x, y_train_y,
                    c="red", s=s, marker="o", alpha=a, label="Training Data")
        plt.scatter(y_test_x, y_test_y,
                    c="navy", s=s, marker="s", alpha=a, label="Testing Data")
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
        if(flag_show):
            plt.show()
        if(flag_save):
            plt.savefig(filename)
        plt.close()

def drawTrTeRf(y_train,y_test,y_rf,flag_show=False,flag_save=False,title=None,filename=None):
    if(flag_show or flag_save):
        plt.figure()
        s = 20
        a = 0.4
        y_train_x = [row[0] for row in y_train]
        y_train_y = [row[1] for row in y_train]
        y_test_x = [row[0] for row in y_test]
        y_test_y = [row[1] for row in y_test]
        y_rf_x = [row[0] for row in y_rf]
        y_rf_y = [row[1] for row in y_rf]
        plt.scatter(y_train_x, y_train_y,
                    c="red", s=s, marker="o", alpha=a, label="Training Data")
        plt.scatter(y_test_x, y_test_y,
                    c="navy", s=20, marker="s", alpha=0.4, label="Testing Data")
        plt.scatter(y_rf_x, y_rf_y,
                    c="c", s=20, marker="^", alpha=0.4, label="Predict Result")
        xmin = min(min(y_train_x),min(y_test_x),min(y_rf_x))
        xmax = max(max(y_train_x),max(y_test_x),min(y_rf_x))
        ymin = min(min(y_train_y),min(y_test_y),min(y_rf_y))
        ymax = max(max(y_train_y),max(y_test_y),min(y_rf_y))
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

def main1():
    # fetch train and test data
    result_tr, result_te = hw4_p1_00.getTrTeListPair("final_2g_gongcan.csv", "final_2g_tr.csv", "final_2g_te.csv", True)
    regr_rf_list = []
    for row_tr in result_tr:
        regr_rf_list.append(getRandomForestRegressorTrained(row_tr[1], row_tr[2]))
        print "!"
    for i in range(len(regr_rf_list)):
        title = "LAC="+str(result_tr[i][0][0])+",CI="+str(result_tr[i][0][1])
        filename = "result/result_p1_2_" + str(i)
        y_rf, score = myRandomForestRegressorFit(regr_rf_list[i], result_te[i][1], result_te[i][2], flag_save=True, title=title, filename=filename)
        print i
    return
def main11():
    # fetch train and test data
    result_tr, result_te = hw4_p1_00.getTrTeListPair("final_2g_gongcan.csv", "final_2g_tr.csv", "final_2g_te.csv", True)
    for i in range(len(result_tr)):
        title = "LAC="+str(result_tr[i][0][0])+",CI="+str(result_tr[i][0][1])
        filename = "result/result_p1_2_2g/result_p1_2_TrTe_" + str(i)
        drawTrTe(result_tr[i][2], result_te[i][2], flag_save=True, title=title, filename=filename)
        print i
    return
def main2():
    # fetch train and test data
    result_tr, result_te = hw4_p1_00.getTrTeListPair("final_2g_gongcan.csv", "final_2g_tr.csv", "final_2g_te.csv", True)
    average_list = []
    for i in range(len(result_tr)):
        dis_list2 = []
        for j in range(10):
            regr_rf = getRandomForestRegressorTrained(result_tr[i][1], result_tr[i][2], random_state=j)
            y_rf, score = myRandomForestRegressorFit(regr_rf, result_te[i][1], result_te[i][2])
            #y_rf_list.append(y_rf)
            dis_list2.append(hw4_p1_00.getDistanceList(y_rf, result_te[i][2]))
        average_col = hw4_p1_00.getAverageList(dis_list2)
        average_col.sort()
        average_list.append([average_col,hw4_p1_00.getMedian(average_col, True)])
        title = "LAC="+str(result_tr[i][0][0])+",CI="+str(result_tr[i][0][1])
        filename = "result/result_p1_2_median_" + str(i)
        drawDistances(average_col,flag_save=True,title=title,filename=filename)
        print filename
    file_out = open("result/result_p1_2_median.csv", "w")
    file_out.write("Number,LAC,CI,Lon,Lat,Median,\r\n")
    for i in range(len(average_list)):
        file_out.write(str(i)+",")
        file_out.write(str(result_tr[i][0][0])+",")
        file_out.write(str(result_tr[i][0][1])+",")
        file_out.write(str(result_tr[i][0][2])+",")
        file_out.write(str(result_tr[i][0][3])+",")
        file_out.write(str(average_list[i][1])+",")
        file_out.write("\r\n")
    file_out.close()
    return
def main3():
    # fetch train and test data
    result_tr, result_te = hw4_p1_00.getTrTeListPair("final_2g_gongcan.csv", "final_2g_tr.csv", "final_2g_te.csv", True)
    average_list2 = []
    dis_list = []
    y_rf = []
    for i in range(len(result_tr)):
        regr_rf = getRandomForestRegressorTrained(result_tr[i][1], result_tr[i][2], random_state=2)
        average_list = []
        average_list2.append(average_list)
        for j in range(len(result_te)):
            y_rf, score = myRandomForestRegressorFit(regr_rf, result_te[j][1], result_te[j][2])
            dis_list = hw4_p1_00.getDistanceList(y_rf, result_te[j][2])
            dis_list.sort()
            average_list.append(hw4_p1_00.getMedian(dis_list, True))
            title = "TrainNo="+str(i)+",TestNo="+str(j)
            filename = "result/result_p1_3/result_p1_3_te" + str(j) + "_tr" + str(i)
            drawDistances(dis_list,flag_save=True,title=title,filename=filename,max_y=0.01)
            print filename
            
    file_out = open("result/result_p1_3_median_list.csv", "w")
    file_out.write("TrNumber,TrLAC,TrCI,TrLon,TrLat,TeNumber,TeLAC,TeCI,TeLon,TeLat,Median,\r\n")
    for i in range(len(average_list2)):
        for j in range(len(average_list)):
            file_out.write(str(i)+",")
            file_out.write(str(result_tr[i][0][0])+",")
            file_out.write(str(result_tr[i][0][1])+",")
            file_out.write(str(result_tr[i][0][2])+",")
            file_out.write(str(result_tr[i][0][3])+",")
            file_out.write(str(j)+",")
            file_out.write(str(result_te[j][0][0])+",")
            file_out.write(str(result_te[j][0][1])+",")
            file_out.write(str(result_te[j][0][2])+",")
            file_out.write(str(result_te[j][0][3])+",")
            file_out.write(str(average_list2[i][j])+",")
            file_out.write("\r\n")
    file_out.close()
    
    file_out = open("result/result_p1_3_median_matrix.csv", "w")
    file_out.write("-,")
    for i in range(len(average_list)):
        file_out.write(str(i)+",")
    file_out.write("\r\n")
    for i in range(len(average_list2)):
        file_out.write(str(i)+",")
        for j in range(len(average_list)):
            file_out.write(str(average_list2[i][j])+",")
        file_out.write("\r\n")
    file_out.close()
    return
def main31():
    # fetch train and test data
    result_tr, result_te = hw4_p1_00.getTrTeListPair("final_2g_gongcan.csv", "final_2g_tr.csv", "final_2g_te.csv", True)
    for i in [36,30,27,38,8,50,44,45,40,48]:
        regr_rf = getRandomForestRegressorTrained(result_tr[i][1], result_tr[i][2], random_state=2)
        for j in [36]:
            y_rf, score = myRandomForestRegressorFit(regr_rf, result_te[j][1], result_te[j][2])
            title = "TrainNo="+str(i)+",TestNo="+str(j)
            filename = "result/result_p1_3/result_p1_3_pos_te" + str(j) + "_tr" + str(i)
            drawTrTeRf(result_tr[i][2],result_te[j][2],y_rf,flag_save=True,title=title,filename=filename)
            print filename
    return
'''
resultTr, resultTe = hw4_p1_00.getTrTeListPair("final_2g_gongcan.csv", "final_2g_tr.csv", "final_2g_te.csv", True)
regr_rf = getRandomForestRegressorTrained(resultTr[0][1], resultTr[0][2])
y_rf, score = myRandomForestRegressorFit(regr_rf, resultTe[0][1], resultTe[0][2], flag_save=True, title="test1", filename="result/test1")
#y_rf, score = myRandomForestRegressorFit(regr_rf, resultTe[0][1], resultTe[0][2], flag_show=True, flag_save=False, title="test2", filename="result/test2")
y_rf, score = myRandomForestRegressorFit(regr_rf, resultTe[0][1], resultTe[0][2], flag_show=False, flag_save=True, title="test3", filename="result/test3")
#y_rf, score = myRandomForestRegressorFit(regr_rf, resultTe[0][1], resultTe[0][2], flag_show=False, flag_save=False, title="test4", filename="result/test4")

print len(y_rf),len(resultTe[0][2])
for i in range(max(len(y_rf),len(resultTe[0][2]))):
    print y_rf[i],resultTe[0][2][i]
'''
main31()
