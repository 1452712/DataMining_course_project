#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:56:32 2017
HMM main part
@author: luminous
"""
import numpy as np
import pandas as pd
import math
from hmmlearn import hmm
from sklearn import preprocessing
#from sklearn.cross_validation import train_test_split

import hw4_p2_00
_minX = 0
_minY = 0
###############################################################
# Common Part
# import data
def initData(filename):
    data = pd.read_csv(filename)
    test_rate = 0.25
    train_data = []
    test_data = []

    # parse data
    for i in range(max(data.PathID) + 1):
        temp = data[data.PathID == i]
        train_temp = []
        test_temp = []
        

        total = len(temp)         
        test_amount = int(math.ceil( float(total) * test_rate))
        # Detect for trasmission
        if test_amount <= 2:
            continue
        test_temp = temp.head(test_amount)
        train_temp = temp.tail(total - test_amount)
        train_data.append(train_temp)
        test_data.append(test_temp)
        """
        test_select = int(math.ceil(1.0 / test_rate))
        if len(temp) <= 2 or test_select >= len(temp):
            continue
        flag = True
        while len(temp) > 0:
            if flag == True:
                test_temp = temp.head(1)
                temp = temp.tail(len(temp) - 1)
                train_amount = min(test_select - 1, len(temp))
                train_temp = temp.head(train_amount)
                temp = temp.tail(len(temp) - train_amount)
                flag = False
            test_temp = np.vstack((test_temp, temp.head(1)))
            temp = temp.tail(len(temp) - 1)
            train_amount = min(test_select - 1, len(temp))
            train_temp = np.vstack((train_temp, temp.head(train_amount)))
            temp = temp.tail(len(temp) - train_amount)
        train_data.append(train_temp)
        test_data.append(test_temp)
        """
        """
        test_select = int(math.ceil(1.0 / test_rate))
        if len(temp) <= 2 or test_select >= len(temp):
            continue
        # divide evenly
        j = 0
        for row in temp:
            if j % test_select == 0:
                test_temp.append(row)
            else:
                train_temp.append(row)
            j += 1
        train_data.append(train_temp)
        test_data.append(test_temp)
        """
    return train_data, test_data

def getMinPos(filename):
    data = pd.read_csv(filename)
    minX = min([data.at[i,"Longitude"] for i in range(len(data))])
    minY = min([data.at[i,"Latitude"] for i in range(len(data))])
    return minX,minY

def getDistance(x,y):
    s = 0
    for i in range(len(x)):
        s += (x[i]-y[i])*(x[i]-y[i])
    return math.sqrt(s)

def getManhattanDistance(x,y):
    s = 0
    for i in range(len(x)):
        s += abs(x[i]-y[i])
    return s
###############################################################

###############################################################
# a) single HMM: based on RSSI
def getHMMTrainDataSingle(data):
    state_0, amount = np.unique(data.Grid_ID, return_counts=True)
    N = len(state_0)
    M = len(data)
    observector = []
    for i in range(M):
        # with cell id
        vector = [data["RSSI_1"].loc[data.index[i]], data["RSSI_2"].loc[data.index[i]],data["RSSI_3"].loc[data.index[i]],
                  data["RSSI_4"].loc[data.index[i]], data["RSSI_5"].loc[data.index[i]],data["RSSI_6"].loc[data.index[i]]]
        observector.append(vector)
    observation = np.array(observector).T
    
    pi = []
    for i in range(N):
        pi.append(amount[i] / float(N))
    startprob = (preprocessing.normalize(pi,norm='l1'))[0]
    
    # According to equation(1)
    a = []
    for i in range(N):
        delta = 0
        denominator = 0
        for r in range(1, M + 1):
            for j in range(1, 6):
                denominator = denominator + 1
                # Kronecker delta function
                cond = observector[r - 1][j] - observector[r - 1][j - 1]
                if cond == i or cond == 0:
                    delta = delta + 1
                else:
                    #magnitude k
                    delta = delta + np.linalg.norm(i)
        a.append(delta / denominator)
    trans = [[0 for col in range(N)] for row in range(N)]
    for i in range(N):
        for j in range(N):
            k = j - i
            if k < 0:
                trans[j][i] = 0
            else:
                trans[j][i] = a[k]
                    
    transmat = np.array(preprocessing.normalize(trans,norm='l1'))
    
    # According to emission score
    ########### TODO: histogram & Gaussian
    lam_match = 3
    d_max = 32
    emm_score = [[0 for col in range(M)] for row in range(N)]
    for i in range(M):
        for j in range(N): #state
            match1 = []
            match2 = []
            state = data[data.Grid_ID == state_0[j]]
            if len(state) <= 0:
                emm_score[i][j] = d_max
                continue
            if data["CellID_1"].loc[data.index[i]] == state["CellID_1"].loc[state.index[0]]:
                match1.append(data["RSSI_1"].loc[data.index[i]])
                match2.append(state["RSSI_1"].loc[state.index[0]])
            if data["CellID_2"].loc[data.index[i]] == state["CellID_2"].loc[state.index[0]]:
                match1.append(data["RSSI_2"].loc[data.index[i]])
                match2.append(state["RSSI_2"].loc[state.index[0]])
            if data["CellID_3"].loc[data.index[i]] == state["CellID_3"].loc[state.index[0]]:
                match1.append(data["RSSI_3"].loc[data.index[i]])
                match2.append(state["RSSI_3"].loc[state.index[0]])
            if data["CellID_4"].loc[data.index[i]] == state["CellID_4"].loc[state.index[0]]:
                match1.append(data["RSSI_4"].loc[data.index[i]])
                match2.append(state["RSSI_4"].loc[state.index[0]])
            if data["CellID_5"].loc[data.index[i]] == state["CellID_5"].loc[state.index[0]]:
                match1.append(data["RSSI_5"].loc[data.index[i]])
                match2.append(state["RSSI_5"].loc[state.index[0]])
            if data["CellID_6"].loc[data.index[i]] == state["CellID_6"].loc[state.index[0]]:
                match1.append(data["RSSI_6"].loc[data.index[i]])
                match2.append(state["RSSI_6"].loc[state.index[0]])
            
            emm_score[j][i] = len(match1) * lam_match + d_max - getDistance(match1,match2)
    emission = np.array(preprocessing.normalize(emm_score,norm='l1'))
    
    return state_0, N, observation, M, transmat, emission, startprob

def testHMMSingle(data, model, grid_gps, state, observation):
    error = []
    predict_gps = []
    actual_gps = []
    
    for i in range(len(data)):
        # no cell id
        vector = [data["RSSI_1"].loc[data.index[i]], data["RSSI_2"].loc[data.index[i]],data["RSSI_3"].loc[data.index[i]],
                  data["RSSI_4"].loc[data.index[i]],data["RSSI_5"].loc[data.index[i]],data["RSSI_6"].loc[data.index[i]]]
        #tester = np.array(vector).T
        #state_sequence = model.predict(tester)
        index = 0
        min_dis = 10000000
        for j in range(len(observation)):
            dis = getDistance(vector, observation[j])
            if dis < min_dis:
                index = j
                min_dis = dis
            
        tester = np.array([index]).T
        logprob, state_sequence = model.decode(tester, algorithm="viterbi")
        pre = grid_gps[grid_gps.Grid_Id == state[state_sequence[0]]]
        #print pre
        pre_gps = [pre["Grid_center_x"].loc[pre.index[0]], pre["Grid_center_y"].loc[pre.index[0]]]
        act_gps = [data["Grid_center_x"].loc[data.index[i]], data["Grid_center_y"].loc[data.index[i]]]
        predict_gps.append(pre_gps)
        actual_gps.append(act_gps)
        error.append(getDistance(pre_gps, act_gps))
    return error, predict_gps, actual_gps

def singleHMM(train_data, test_data, label):
    grid_gps = pd.read_csv("data/final_grid_id_sorted.csv")
    error = []
    # train by group
    for i in range(len(train_data)):
        print "Current training data: route ", i
        # model
        state, N, observation, M, transmat, emission, startprob = getHMMTrainDataSingle(train_data[i])
        #print state
        ########### TODO: Change Parameter
        model = hmm.MultinomialHMM(n_components=N, startprob_prior=1.0, transmat_prior=1.0, algorithm='viterbi')
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.emissionprob_ = emission
        # train
        ########### TODO: decode?
        #model.fit(observation)
        
        # test
        error, predict_gps, actual_gps = testHMMSingle(test_data[i], model, grid_gps,state, observation)
        # draw
        title = "Data Source: " + str(label) +"; Route: " + str(i)
        filename = "result_p2/single/singleHMM_" + str(label) + "_" + str(i)
        hw4_p2_00.drawPreAct(predict_gps, actual_gps, True, True, title, filename)    
        
    print "Single HMM is done."
    print
    
    return error
###############################################################

###############################################################
# b) double HMM: based on fingerprint
def getHMMTrainDataDouble(data):
    M = len(data)
    observector = []
    state_cal = []
    for i in range(M):
        # with cell id
        state_cal.append([data["Cell_ID_x"].loc[data.index[i]], data["Cell_ID_y"].loc[data.index[i]]])
        vector = [data["RSSI_1"].loc[data.index[i]], data["RSSI_2"].loc[data.index[i]],data["RSSI_3"].loc[data.index[i]],
                  data["RSSI_4"].loc[data.index[i]], data["RSSI_5"].loc[data.index[i]],data["RSSI_6"].loc[data.index[i]]]
        observector.append(vector)
    observation = np.array(observector).T
    
    state_0 = []
    amount = []
    for i in range(M):
        if state_cal[i] in state_0:
            p = state_0.index(state_cal[i])
            amount[p] += 1
        else:
            state_0.append(state_cal[i])
            amount.append(1)
    N = len(state_0)
    
    pi = []
    for i in range(N):
        pi.append(amount[i] / float(N))
    startprob = (preprocessing.normalize(pi,norm='l1'))[0]
    
    # According to Manhattan distance
    trans = [[0 for col in range(N)] for row in range(N)]
    for i in range(N):
        for j in range(N):
            if state_0[j][0] == state_0[i][0] and state_0[j][1] == state_0[i][1]:
                trans[j][i] = 1
            else:
                trans[j][i] = float(1) / getManhattanDistance(state_0[i], state_0[j])
                    
    transmat = np.array(preprocessing.normalize(trans,norm='l1'))
    
    # According to emission score
    lam_match = 3
    d_max = 1
    emm_score = [[0 for col in range(M)] for row in range(N)]
    for i in range(M):
        for j in range(N): #state
            match1 = []
            match2 = []
            state = data[(data.Cell_ID_x == state_0[j][0]) &
                         (data.Cell_ID_y == state_0[j][1])]
            if len(state) <= 0:
                emm_score[i][j] = d_max
                continue
            if data["CellID_1"].loc[data.index[i]] == state["CellID_1"].loc[state.index[0]]:
                match1.append(data["RSSI_1"].loc[data.index[i]])
                match2.append(state["RSSI_1"].loc[state.index[0]])
            if data["CellID_2"].loc[data.index[i]] == state["CellID_2"].loc[state.index[0]]:
                match1.append(data["RSSI_2"].loc[data.index[i]])
                match2.append(state["RSSI_2"].loc[state.index[0]])
            if data["CellID_3"].loc[data.index[i]] == state["CellID_3"].loc[state.index[0]]:
                match1.append(data["RSSI_3"].loc[data.index[i]])
                match2.append(state["RSSI_3"].loc[state.index[0]])
            if data["CellID_4"].loc[data.index[i]] == state["CellID_4"].loc[state.index[0]]:
                match1.append(data["RSSI_4"].loc[data.index[i]])
                match2.append(state["RSSI_4"].loc[state.index[0]])
            if data["CellID_5"].loc[data.index[i]] == state["CellID_5"].loc[state.index[0]]:
                match1.append(data["RSSI_5"].loc[data.index[i]])
                match2.append(state["RSSI_5"].loc[state.index[0]])
            if data["CellID_6"].loc[data.index[i]] == state["CellID_6"].loc[state.index[0]]:
                match1.append(data["RSSI_6"].loc[data.index[i]])
                match2.append(state["RSSI_6"].loc[state.index[0]])
            emm_score[j][i] = len(match1) * lam_match + d_max - getDistance(match1,match2)
            
    emission = np.array(preprocessing.normalize(emm_score,norm='l1'))
    
    return state_0, N, observation, M, transmat, emission, startprob

def get2ndHMMTrainDataDouble(data):
    M = len(data)
    observector = []
    state_cal = []
    for i in range(M):
        # with cell id
        state_cal.append([data["Cell_center_x"].loc[data.index[i]], data["Cell_center_y"].loc[data.index[i]]])
        vector = [data["Cell_ID_x"].loc[data.index[i]], data["Cell_ID_y"].loc[data.index[i]]]
        observector.append(vector)
    observation = np.array(observector).T
    
    state_0 = []
    amount = []
    for i in range(M):
        if state_cal[i] in state_0:
            p = state_0.index(state_cal[i])
            amount[p] += 1
        else:
            state_0.append(state_cal[i])
            amount.append(1)
    N = len(state_0)
    
    pi = []
    for i in range(N):
        pi.append(amount[i] / float(N))
    startprob = (preprocessing.normalize(pi,norm='l1'))[0]
    
    # According to Manhattan distance
    d_max = 1
    trans = [[0 for col in range(N)] for row in range(N)]
    for i in range(N):
        for j in range(N):
            if state_0[j][0] == state_0[i][0] and state_0[j][1] == state_0[i][1]:
                trans[j][i] = 1
            else:
                trans[j][i] = d_max - getManhattanDistance(state_0[i], state_0[j])
                    
    transmat = np.array(preprocessing.normalize(trans,norm='l1'))
    
    # Self-define emission score
    emm_score = [[0 for col in range(M)] for row in range(N)]
    for i in range(M):
        for j in range(N):
            state = data[(data.Cell_ID_x == state_0[j][0]) &
                         (data.Cell_ID_y == state_0[j][1])]
            if len(state) <= 0:
                emm_score[j][i] = 0
            else:
                emm_score[j][i] = float(1) / getManhattanDistance([data["Cell_center_x"].loc[data.index[i]], data["Cell_center_y"].loc[data.index[i]]], 
                                                     [state["Cell_center_x"].loc[state.index[0]], state["Cell_center_y"].loc[state.index[0]]])
    emission = np.array(preprocessing.normalize(emm_score,norm='l1'))
    
    return state_0, N, observation, M, transmat, emission, startprob

def testHMMDouble(data, model, model_2, grid_gps,state_1,state_2, observation):
    error = []
    predict_gps = []
    actual_gps = []
    
    for i in range(len(data)):
        # no cell id
        vector = [data["RSSI_1"].loc[data.index[i]], data["RSSI_2"].loc[data.index[i]],data["RSSI_3"].loc[data.index[i]],
                  data["RSSI_4"].loc[data.index[i]],data["RSSI_5"].loc[data.index[i]],data["RSSI_6"].loc[data.index[i]]]
        # tester = np.array(vector)
        #tester = np.array(data["Grid_ID"].loc[data.index[i]])
        index = 0
        min_dis = 10000000
        for j in range(len(observation)):
            dis = getDistance(vector, observation[j])
            if dis < min_dis:
                index = j
                min_dis = dis
        
        tester = np.array([index])
        state_sequence = state_1[model.predict(tester)]
        pre_gps = [_minX + (state_sequence[0] + 0.5)*0.0005,_minY + (state_sequence[1] + 0.5)*0.0005]
        act_gps = [data["Grid_center_x"].loc[data.index[i]], data["Grid_center_y"].loc[data.index[i]]]
        predict_gps.append(pre_gps)
        actual_gps.append(act_gps)
        error.append(getDistance(pre_gps, act_gps))
    return error, predict_gps, actual_gps

def doubleHMM(train_data, test_data, label):
    grid_gps = pd.read_csv("data/final_grid_id_sorted.csv")
    error = []
    # train by group
    for i in range(len(train_data)):
        print "Current training data: route ", i
        # 1st model
        state_1, N, observation, M, transmat, emission, startprob = getHMMTrainDataDouble(train_data[i])
        ########### TODO: Change Parameter
        model = hmm.MultinomialHMM(n_components=N, startprob_prior=1.0, transmat_prior=1.0, algorithm='viterbi')
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.emissionprob_ = emission
        # train
        # model.fit(observation)
        
        # 2nd model
        state_2, N_2, observation_2, M_2, transmat_2, emission_2, startprob_2 = get2ndHMMTrainDataDouble(train_data[i])
        ########### TODO: Change Parameter
        model_2 = hmm.MultinomialHMM(n_components=N_2, startprob_prior=1.0, transmat_prior=1.0, algorithm='viterbi')
        model_2.startprob_ = startprob_2
        model_2.transmat_ = transmat_2
        model_2.emissionprob_ = emission_2
        # train
        # model_2.fit(observation)
        
        #test
        error, predict_gps, actual_gps = testHMMDouble(train_data[i], model, model_2, grid_gps, state_1, state_2, observation)
        
        # draw
        title = "Data Source: " + str(label) +"; Route: " + str(i)
        filename = "result_p2/double/double_HMM_" + str(label) + "_" + str(i)
        hw4_p2_00.drawPreAct(predict_gps, actual_gps, True, True, title, filename)
        
    print "Double HMM is done."
    print
    
    return error
###############################################################
# main
def main():
    files = ["my_final_2g_tr.csv", "my_final_2g_tr_2.csv", "my_final_4g_tr.csv", "my_final_4g_tr_2.csv"]
    for i in range(len(files)):
        # import data
        train_data, test_data = initData(files[i])
        # compare two type of HMM
        # you can choose train_data as test_data
        single_error = singleHMM(train_data, test_data, i)
        double_error = doubleHMM(train_data, test_data, i)    
        # draw
        title = "Data Source: " + files[i]
        filename = "result_p2/error_compare_hmm_" + str(i)
        hw4_p2_00.drawError(single_error, double_error, True, True, title, filename)
        
    return

###############################################################
# Start_up
# Get Global Parameters
_minX,_minY = getMinPos("my_final_2g_tr.csv")
main()

# use the function directly
#train_data, test_data = initData("my_final_2g_tr.csv")
#single_error = doubleHMM(train_data, test_data, 1)