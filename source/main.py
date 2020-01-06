# -*- coding: utf-8 -*-
# to run on server
import sys
sys.path.append("/home/harada/Documents/WorkSpace/adain")
sys.path.append("/home")

import pickle
import time
import random
import torch
import optuna
import math
import numpy as np
import pandas as pd
# from my library
from source.func import makeDataset
from source.func import makeCityData
from source.func import makeTrainData
from source.func import makeTestData
from source.func import makeTestData_sampled
from source.func import objective_ADAIN
from source.func import objective_HARADA
from source.func import objective_FNN
from source.func import evaluate_ADAIN
from source.func import evaluate_HARADA
from source.func import evaluate_FNN
from source.func import evaluate_KNN
from source.func import evaluate_LI
from source.utility import get_dist_angle
from source.utility import MMD
from source.utility import MMD_preComputed

def expAAAI(TRIAL, CITY):

    '''
    AAAI 2018の再実験
    '''

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):

        start = time.time()
        print("----------------")
        print("* SOURCE: {}{}".format(CITY, str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/AAAI18/train_{}{}".format(CITY, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/AAAI18/test_{}{}".format(CITY, str(loop)), fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective_ADAIN, n_trials=TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/AAAI18_{}{}.pkl".format(CITY, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/AAAI18_{}{}.csv".format(CITY, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/AAAI18_{}{}.pkl".format(CITY, str(loop)), "rb"))

        # evaluate
        print("* TARGET: {}{}".format(CITY, str(loop)))
        rmse, accuracy = evaluate_ADAIN(model_state_dict)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hours]".format(str((time.time()-start)/(60*60))))

    with open("result/AAAI18_{}.csv".format(CITY), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{}\n".format(CITY))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), str(rmse_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(rmse_list)))))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{}\n".format(CITY))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), str(accuracy_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(accuracy_list)))))

def expDistance(TRIAL, SOURCE, TARGETs):

    '''
    ある都市で訓練したモデルを他の都市でテスト
    '''

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):

        start = time.time()
        print("*----------------")
        print("*SOURCE: {}{}".format(SOURCE, str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Train1/train_{}{}".format(SOURCE, SOURCE, str(loop)), fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective_ADAIN, n_trials=TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/{}Train1{}.pkl".format(SOURCE, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/{}Train{}1.csv".format(SOURCE, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/{}Train{}1.pkl".format(SOURCE, str(loop)), "rb"))

        # to evaluate
        rmse_tmp = list()
        accuracy_tmp = list()

        # save dataset path
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Train1/test_{}{}".format(SOURCE, SOURCE, str(loop)), fp)

        # evaluate
        print("*TARGET: {}{}".format(SOURCE, str(loop)))
        rmse, accuracy = evaluate_ADAIN(model_state_dict)
        rmse_tmp.append(rmse)
        accuracy_tmp.append(accuracy)

        for TARGET in TARGETs:

            # save dataset path
            with open("tmp/testPath.pkl", "wb") as fp:
                pickle.dump("dataset/{}Train1/test_{}{}".format(SOURCE, TARGET, str(loop)), fp)

            # evaluate
            print("*TARGET: {} ---".format(TARGET))
            rmse, accuracy = evaluate_ADAIN(model_state_dict)
            rmse_tmp.append(rmse)
            accuracy_tmp.append(accuracy)

        rmse_list.append(rmse_tmp)
        accuracy_list.append(accuracy_tmp)
        print("time = {} [hours]".format(str((time.time() - start) / (60 * 60))))

    # to output
    rmse = np.average(np.array(rmse_list), axis=0)
    rmse = list(map(lambda x: str(x), rmse))
    tmp = rmse_list.copy()
    rmse_list = []
    for tmp_i in tmp:
        rmse_list.append(list(map(lambda x: str(x), tmp_i)))

    accuracy = np.average(np.array(accuracy_list), axis=0)
    accuracy = list(map(lambda x: str(x), accuracy))
    tmp = accuracy_list.copy()
    accuracy_list = []
    for tmp_i in tmp:
        accuracy_list.append(list(map(lambda x: str(x), tmp_i)))

    c = pd.read_csv("rawdata/zheng2015/city.csv", index_col="name_english")
    lat_local = c.at[SOURCE, "latitude"]
    lon_local = c.at[SOURCE, "longitude"]
    distance = list()
    for TARGET in TARGETs:
        lat = c.at[TARGET, "latitude"]
        lon = c.at[TARGET, "longitude"]
        result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
        distance.append(str(result["distance"]/1000.0))

    with open("result/{}Train1.csv".format(SOURCE), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{},{}\n".format(SOURCE, ",".join(TARGETs)))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), ",".join(rmse_list[loop])))
        result.write("average,{}\n".format(",".join(rmse)))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{},{}\n".format(SOURCE, ",".join(TARGETs)))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), ",".join(accuracy_list[loop])))
        result.write("average,{}\n".format(",".join(accuracy)))
        result.write("--------------------------------------------\n")
        result.write("Distance\n")
        result.write("----------\n")
        result.write("distance,0.0,{}\n".format(",".join(distance)))

def expMAX(TRIAL, SOURCE, TARGETs):

    '''
    一つの都市の全てのデータで訓練したモデルを他の都市でテスト
    '''

    start = time.time()
    print("*----------------")
    print("*SOURCE: {}".format(SOURCE))

    # save dataset path
    with open("tmp/trainPath.pkl", "wb") as fp:
        pickle.dump("dataset/{}Train/train_{}Max".format(SOURCE, SOURCE), fp)

    # training & parameter tuning by optuna
    # -- activate function, optimizer, eopchs, batch size
    study = optuna.create_study()
    study.optimize(objective_ADAIN, n_trials=TRIAL)

    # save best model
    model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
    pickle.dump(model_state_dict, open("model/{}TrainMax.pkl".format(SOURCE), "wb"))

    # save train log
    log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
    log.to_csv("log/{}TrainMax.csv".format(SOURCE), index=False)

    # load best model
    model_state_dict = pickle.load(open("model/{}TrainMax.pkl".format(SOURCE), "rb"))

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for TARGET in TARGETs:
        # save dataset path
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Train/test_{}".format(SOURCE, TARGET), fp)

        # evaluate
        print("*TARGET: {}".format(TARGET))
        rmse, accuracy = evaluate_ADAIN(model_state_dict)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

    print("time = {} [hours]".format(str((time.time() - start) / (60 * 60))))

    # to output
    c = pd.read_csv("rawdata/zheng2015/city.csv", index_col="name_english")
    lat_local = c.at[SOURCE, "latitude"]
    lon_local = c.at[SOURCE, "longitude"]
    distance = list()
    for TARGET in TARGETs:
        lat = c.at[TARGET, "latitude"]
        lon = c.at[TARGET, "longitude"]
        result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
        distance.append(str(result["distance"]/1000.0))

    with open("result/{}TrainMax.csv".format(SOURCE), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{},{}\n".format(SOURCE, ",".join(TARGETs)))
        result.write("max,NULL,{}\n".format(",".join(rmse_list)))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{},{}\n".format(SOURCE, ",".join(TARGETs)))
        result.write("max,NULL,{}\n".format(",".join(accuracy_list)))
        result.write("--------------------------------------------\n")
        result.write("Distance\n")
        result.write("----------\n")
        result.write("distance,0.0,{}\n".format(",".join(distance)))

def exp19cities(TRIAL, TARGET):

    '''
    19都市で訓練したモデルをでテスト
    '''

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):
        start = time.time()
        print("----------------")
        print("* SOURCE: All{}".format(str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19/train{}".format(TARGET, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19/test{}".format(TARGET, str(loop)), fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective_ADAIN, n_trials=TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/{}Test19_{}.pkl".format(TARGET, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/{}Test19_{}.csv".format(TARGET, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/{}Test19_{}.pkl".format(TARGET, str(loop)), "rb"))

        # evaluate
        print("* TARGET: {}{}".format(TARGET, str(loop)))
        rmse, accuracy = evaluate_ADAIN(model_state_dict)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hours]".format(str((time.time() - start) / (60 * 60))))

    with open("result/{}Test19.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), str(rmse_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(rmse_list)))))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), str(accuracy_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(accuracy_list)))))

def expProposal(TRIAL, TARGET):

    '''
    提案手法
    '''

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(2, 3):
        start = time.time()
        print("----------------")
        print("* SOURCE: All{}".format(str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19_city/train{}".format(TARGET, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19_city/test{}".format(TARGET, str(loop)), fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective_HARADA, n_trials=TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/{}Test19_city_{}.pkl".format(TARGET, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/{}Test19_city_{}.csv".format(TARGET, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/{}Test19_city_{}.pkl".format(TARGET, str(loop)), "rb"))

        # evaluate
        print("* TARGET: {}{}".format(TARGET, str(loop)))
        rmse, accuracy = evaluate_HARADA(model_state_dict)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hours]".format(str((time.time() - start) / (60 * 60))))

    with open("result/{}Test19_city.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), str(rmse_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(rmse_list)))))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), str(accuracy_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(accuracy_list)))))

def expFNN(TRIAL, TARGET):

    '''
    19都市で訓練したモデルをでテスト (FNN)
    '''

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):
        start = time.time()
        print("----------------")
        print("* SOURCE: All{}".format(str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19/train{}".format(TARGET, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19/test{}".format(TARGET, str(loop)), fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective_FNN, n_trials=TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/{}Test19FNN_{}.pkl".format(TARGET, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/{}Test19FNN_{}.csv".format(TARGET, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/{}Test19FNN_{}.pkl".format(TARGET, str(loop)), "rb"))

        # evaluate
        print("* TARGET: {}{}".format(TARGET, str(loop)))
        rmse, accuracy = evaluate_FNN(model_state_dict)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hours]".format(str((time.time() - start) / (60 * 60))))

    with open("result/{}Test19FNN.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), str(rmse_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(rmse_list)))))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), str(accuracy_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(accuracy_list)))))

def expKNN(TARGET):

    '''
    KNN: 距離の近い K このステーションの平
    '''

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):
        start = time.time()
        print("----------------")

        # save dataset path
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19/test{}".format(TARGET, str(loop)), fp)

        # evaluate
        print("* TARGET: {}{}".format(TARGET, str(loop)))
        rmse, accuracy = evaluate_KNN(K=3)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hours]".format(str((time.time() - start) / (60 * 60))))

    with open("result/{}Test19KNN.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), str(rmse_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(rmse_list)))))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), str(accuracy_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(accuracy_list)))))

def expLI(TARGET):

    '''
    LI: 距離の逆数の比で重み付け
    '''

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):
        start = time.time()
        print("----------------")

        # save dataset path
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19/test{}".format(TARGET, str(loop)), fp)

        # evaluate
        print("* TARGET: {}{}".format(TARGET, str(loop)))
        rmse, accuracy = evaluate_LI()
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hours]".format(str((time.time() - start) / (60 * 60))))

    with open("result/{}Test19LI.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), str(rmse_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(rmse_list)))))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), str(accuracy_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(accuracy_list)))))

def exp5cities(TRIAL, TARGET):

    '''
    距離の近い5都市で訓練したモデルをでテスト
    '''

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):
        start = time.time()
        print("----------------")
        print("* SOURCE: Dist-{}".format(str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test5/train{}".format(TARGET, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test5/test{}".format(TARGET, str(loop)), fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective_ADAIN, n_trials=TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/{}Test5_{}.pkl".format(TARGET, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/{}Test5_{}.csv".format(TARGET, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/{}Test5_{}.pkl".format(TARGET, str(loop)), "rb"))

        # evaluate
        print("* TARGET: {}{}".format(TARGET, str(loop)))
        rmse, accuracy = evaluate_ADAIN(model_state_dict)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hours]".format(str((time.time() - start) / (60 * 60))))

    with open("result/{}Test5.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), str(rmse_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(rmse_list)))))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), str(accuracy_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(accuracy_list)))))

def exp1city(TRIAL, SOURCEs, TARGET):

    '''
    他の都市で訓練したモデルでテスト
    '''

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):

        start = time.time()

        # to evaluate
        rmse_tmp = list()
        accuracy_tmp = list()

        for SOURCE in SOURCEs:

            print("*----------------")
            print("*SOURCE: {}{}".format(SOURCE, str(loop)))

            # save dataset path
            with open("tmp/testPath.pkl", "wb") as fp:
                pickle.dump("dataset/{}Test1/test_{}{}".format(TARGET, SOURCE, str(loop)), fp)
            with open("tmp/trainPath.pkl", "wb") as fp:
                pickle.dump("dataset/{}Test1/train_{}{}".format(TARGET, SOURCE, str(loop)), fp)

            # training & parameter tuning by optuna
            # -- activate function, optimizer, eopchs, batch size
            study = optuna.create_study()
            study.optimize(objective_ADAIN, n_trials=TRIAL)

            # save best model
            model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
            pickle.dump(model_state_dict, open("model/{}Test1_{}{}.pkl".format(TARGET, SOURCE, str(loop)), "wb"))

            # save train log
            log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
            log.to_csv("log/{}Test1_{}.csv".format(TARGET, SOURCE, str(loop)), index=False)

            # load best model
            model_state_dict = pickle.load(open("model/{}Test1_{}{}.pkl".format(TARGET, SOURCE, str(loop)), "rb"))

            # evaluate
            print("*TARGET: {}{}".format(TARGET, str(loop)))
            rmse, accuracy = evaluate_ADAIN(model_state_dict)
            rmse_tmp.append(rmse)
            accuracy_tmp.append(accuracy)

        rmse_list.append(rmse_tmp)
        accuracy_list.append(accuracy_tmp)
        print("time = {} [hours]".format(str((time.time() - start) / (60 * 60))))

    # to output
    rmse = np.average(np.array(rmse_list), axis=0)
    rmse = list(map(lambda x: str(x), rmse))
    tmp = rmse_list.copy()
    rmse_list = []
    for tmp_i in tmp:
        rmse_list.append(list(map(lambda x: str(x), tmp_i)))

    accuracy = np.average(np.array(accuracy_list), axis=0)
    accuracy = list(map(lambda x: str(x), accuracy))
    tmp = accuracy_list.copy()
    accuracy_list = []
    for tmp_i in tmp:
        accuracy_list.append(list(map(lambda x: str(x), tmp_i)))

    c = pd.read_csv("rawdata/zheng2015/city.csv", index_col="name_english")
    lat_local = c.at[TARGET, "latitude"]
    lon_local = c.at[TARGET, "longitude"]
    distance = list()
    for SOURCE in SOURCEs:
        lat = c.at[SOURCE, "latitude"]
        lon = c.at[SOURCE, "longitude"]
        result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
        distance.append(str(result["distance"]/1000.0))

    with open("result/{}Test1.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{},{}\n".format(TARGET, ",".join(SOURCEs)))
        for loop in range(len(rmse_list)):
            result.write("exp{},NULL,{}\n".format(str(loop), ",".join(rmse_list[loop])))
        result.write("average,NULL,{}\n".format(",".join(rmse)))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{},{}\n".format(TARGET, ",".join(SOURCEs)))
        for loop in range(len(accuracy_list)):
            result.write("exp{},NULL,{}\n".format(str(loop), ",".join(accuracy_list[loop])))
        result.write("average,NULL,{}\n".format(",".join(accuracy)))
        result.write("--------------------------------------------\n")
        result.write("Distance\n")
        result.write("----------\n")
        result.write("distance,0.0,{}\n".format(",".join(distance)))

def AAAI18():

    CITY = "BeiJing"
    dataset = "AAAI18"
    S = list(pd.read_csv("database/station/station_{}.csv".format(CITY), dtype=object)["sid"])
    trainNum = math.floor(len(S)*0.67)
    testNum = len(S) - trainNum

    print("stationNum: {}".format(str(len(S))))
    print("trainNum: {}".format(str(trainNum)))
    print("testNum: {}".format(str(testNum)))
    print("--------------")

    for loop in range(1, 4):

        print("* Shuffle Loop: {}".format(str(loop)))
        station = S.copy()
        random.shuffle(station)
        station_train = station[:trainNum]
        station_test = station[trainNum:]

        print("* train set")
        savePath = "dataset/{}/train_{}{}".format(dataset, CITY, str(loop))
        makeTrainData(savePath, station_train)

        print("* test set")
        savePath = "dataset/{}/test_{}{}".format(dataset, CITY, str(loop))
        makeTestData(savePath, station_test, station_train)

def city1train(CITIEs20, CITIEs4):

    print("trainNum: 5")
    print("testNum: 5")
    print("--------------")

    for TARGET in CITIEs4:

        dataset = "{}Train1".format(TARGET)
        print("* Target: {}".format(TARGET))

        for loop in range(1, 4):

            print("* Shuffle Loop: {}".format(str(loop)))
            station_target = list(pd.read_csv("database/station/station_{}.csv".format(TARGET), dtype=object)["sid"])
            random.shuffle(station_target)
            station_train = station_target[:5]

            print("* train set")
            savePath = "dataset/{}/train_{}{}".format(dataset, TARGET, str(loop))
            makeTrainData(savePath, station_train)

            for SOURCE in CITIEs20:

                print("* Source: {}".format(SOURCE))

                if SOURCE == TARGET:
                    station_test = station_target[5:10]
                else:
                    station_test = list(pd.read_csv("database/station/station_{}.csv".format(SOURCE), dtype=object)["sid"])
                    random.shuffle(station_test)
                    station_test = station_test[:5]

                print("* test set")
                savePath = "dataset/{}/test_{}{}".format(dataset, SOURCE, str(loop))
                makeTestData(savePath, station_test, station_train)

def city1test(CITIEs20, CITIEs4):

    print("trainNum: 5")
    print("testNum: 5")
    print("--------------")

    for TARGET in CITIEs4:

        SOURCEs = CITIEs20.copy()
        SOURCEs.remove(TARGET)

        dataset = "{}Test1".format(TARGET)
        print("* Target: {}".format(TARGET))

        for loop in range(1, 4):

            print("* Shuffle Loop: {}".format(str(loop)))
            station_test = list(pd.read_csv("database/station/station_{}.csv".format(TARGET), dtype=object)["sid"])
            random.shuffle(station_test)
            station_test = station_test[:5]

            for SOURCE in SOURCEs:

                print("* Source: {}".format(SOURCE))

                station_train = list(pd.read_csv("database/station/station_{}.csv".format(SOURCE), dtype=object)["sid"])
                random.shuffle(station_train)
                station_train = station_train[:5]

                print("* train set")
                savePath = "dataset/{}/train_{}{}".format(dataset, SOURCE, str(loop))
                makeTrainData(savePath, station_train)

                print("* test set")
                savePath = "dataset/{}/test_{}{}".format(dataset, SOURCE, str(loop))
                makeTestData(savePath, station_test, station_train)

def cityTest5(CITIEs4):

    print("trainNum: 25")
    print("testNum: 5")
    print("--------------")

    for TARGET in CITIEs4:

        print("*---TARGET: {}".format(TARGET))

        if TARGET == "BeiJing":
            SOURCEs = ["LangFang", "TianJin", "BaoDing", "TangShan", "ZhangJiaKou"]
        elif TARGET == "TianJin":
            SOURCEs = ["LangFang", "CangZhou", "TangShan", "BeiJing", "BaoDing"]
        elif TARGET == "ShenZhen":
            SOURCEs = ["XiangGang", "DongGuan", "HuiZhou", "JiangMen", "GuangZhou"]
        else:
            SOURCEs = ["FoShan", "DongGuan", "JiangMen", "ShenZhen", "HuiZhou"]

        dataset = "{}Test5".format(TARGET)

        station_test = list(pd.read_csv("database/station/station_{}.csv".format(TARGET), dtype=object)["sid"])
        random.shuffle(station_test)
        station_test = station_test[:5]

        for loop in range(1, 4):

            print("* Shuffle Loop: {}".format(str(loop)))

            station_train = list()
            for SOURCE in SOURCEs:
                station_source = list(pd.read_csv("database/station/station_{}.csv".format(SOURCE), dtype=object)["sid"])
                random.shuffle(station_source)
                station_train += station_source[:5]
            random.shuffle(station_train)

            print("* train set")
            savePath = "dataset/{}/train{}".format(dataset, str(loop))
            makeTrainData(savePath, station_train)

            print("* test set")
            savePath = "dataset/{}/test{}".format(dataset, str(loop))
            makeTestData(savePath, station_test, station_train)

def cityTest19(CITIEs20, CITIEs4):

    print("trainNum: 95")
    print("testNum: 5")
    print("--------------")

    for TARGET in CITIEs4:

        print("*---TARGET: {}".format(TARGET))

        SOURCEs = CITIEs20.copy()
        SOURCEs.remove(TARGET)
        dataset = "{}Test19".format(TARGET)

        station_test = list(pd.read_csv("database/station/station_{}.csv".format(TARGET), dtype=object)["sid"])
        random.shuffle(station_test)
        station_test = station_test[:5]

        for loop in range(1, 4):

            print("* Shuffle Loop: {}".format(str(loop)))

            station_train = list()
            for SOURCE in SOURCEs:
                station_source = list(pd.read_csv("database/station/station_{}.csv".format(SOURCE), dtype=object)["sid"])
                random.shuffle(station_source)
                station_train += station_source[:5]
            random.shuffle(station_train)

            print("* train set")
            savePath = "dataset/{}/train{}".format(dataset, str(loop))
            makeTrainData(savePath, station_train)

            print("* test set")
            savePath = "dataset/{}/test{}".format(dataset, str(loop))
            makeTestData_sampled(savePath, station_test, station_train)

def cityTest19_cityData(CITIEs4):

    print("trainNum: 95")
    print("testNum: 5")
    print("--------------")

    for TARGET in CITIEs4:

            print("*---TARGET: {}".format(TARGET))

            for loop in range(3, 4):
                print("* Shuffle Loop: {}".format(str(loop)))

                # train data
                dataPath_train = "dataset/{}Test19/train{}".format(TARGET, str(loop))
                savePath_train = "dataset/{}Test19_city/train{}".format(TARGET, str(loop))
                # test data
                dataPath_test = "dataset/{}Test19/test{}".format(TARGET, str(loop))
                savePath_test = "dataset/{}Test19_city/test{}".format(TARGET, str(loop))
                makeCityData(dataPath_train, savePath_train, dataPath_test, savePath_test)

def analysisKNN(TARGET):

    '''
    KNN: 距離の近い K このステーションの平
    '''

    from source.func import analysis_KNN
    from source.func import repeat_KNN

    for loop in range(1, 3):
        print("----------------")
        print("* TARGET: {}{}".format(TARGET, str(loop)))

        # save dataset path
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19/test{}".format(TARGET, str(loop)), fp)

        analysis_KNN(K=3)

    # ---------------------------------------------------------------------- #
    #
    # loop = 1
    # print("----------------")
    # print("* TARGET: {}{}".format(TARGET, str(loop)))
    #
    # # save dataset path
    # with open("tmp/testPath.pkl", "wb") as fp:
    #     pickle.dump("dataset/{}Test19/test{}".format(TARGET, str(loop)), fp)
    #
    # # repeat KNN
    # with open("result/{}Test19KNN_analysis.csv".format(TARGET), "w") as result:
    #     result.write("--------------------------------------------\n")
    #     result.write("RMSE\n")
    #     result.write("----------\n")
    #     result.write("K,RMSE\n")
    #     repeat_KNN(TARGET, 1, 96)

if __name__ == "__main__":

    device = "gpu" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))

    '''
    global parameters
    '''
    ATTRIBUTE = "pm25"
    LSTM_DATA_WIDTH = 24
    TIMEPERIOD = 24 * 30 * 6
    TRIAL = 1

    '''
    20 cities
    '''
    # #気象データが全部Nullの都市は無視
    # CITIEs20 = list()
    # for city in list(pd.read_csv("rawdata/zheng2015/city.csv")["name_english"]):
    #     with open("database/station/station_"+city+".csv", "r") as infile:
    #         infile = infile.readlines()[1:] # 1行目を無視
    #         if len(infile) >= 5:
    #             CITIEs20.append(city)
    # CITIEs20.remove("JiNan")
    # CITIEs20.remove("HeYuan")
    # CITIEs20.remove("JieYang")
    # CITIEs20.remove("ShaoGuan")
    # CITIEs20.remove("DaTong")
    # CITIEs20.remove("DeZhou")
    # CITIEs20.remove("BinZhou")
    # CITIEs20.remove("DongYing")
    # CITIEs20.remove("ChenZhou")
    #
    # with open("tmp/station_table.csv", "w") as outfile:
    #     outfile.write("都市,観測ステーション数\n")
    #     for city in CITIEs20:
    #         with open("database/station/station_{}.csv".format(city), "r") as infile:
    #             outfile.write("{},{}\n".format(city, str(len(infile.readlines())-1)))

    '''
    4 cities
    '''
    #CITIEs4 = ["BeiJing", "TianJin", "ShenZhen", "GuangZhou"]

    '''
    create dataset
    '''
    #makeDataset(CITIEs20, ATTRIBUTE, LSTM_DATA_WIDTH, TIMEPERIOD)
    CITIEs4 = ["TianJin"]
    cityTest19_cityData(CITIEs4)
    #cityTest19(CITIEs20, CITIEs4)
    #cityTest5(CITIEs4)
    #city1test(CITIEs20, CITIEs4)
    #city1train(CITIEs20, CITIEs4)
    #AAAI18()

    '''
    Experiment0:
    AAAI'18 の再実験
    # '''
    # CITY = "BeiJing"
    # expAAAI(TRIAL, CITY)

    '''
    Experiment1:
    物理距離と特徴距離の仮説検証実験
    '''
    # for SOURCE in CITIEs4:
    #     TARGETs = CITIEs20.copy()
    #     TARGETs.remove(SOURCE)
    #     expDistance(TRIAL, SOURCE, TARGETs)

    '''
    Experiment2:
    最大性能を検証するための実験
    '''
    # for SOURCE in CITIEs4:
    #     TARGETs = CITIEs20.copy()
    #     TARGETs.remove(SOURCE)
    #     expMAX(TRIAL, SOURCE, TARGETs)

    '''
    Experiment3:
    全都市で訓練したモデルの性能検証実験
    '''
    # for TARGET in CITIEs4:
    #     exp19cities(TRIAL, TARGET)

    '''
    Experiment4:
    距離の近い都市で訓練したモデルの性能検証実験
    '''
    # CITIEs4 = ["GuangZhou"]
    # for TARGET in CITIEs4:
    #     exp5cities(TRIAL, TARGET)

    '''
    Experiment5:
    他都市で訓練したモデルでの性能検証実験
    '''
    # for TARGET in CITIEs4:
    #     SOURCEs = CITIEs20.copy()
    #     SOURCEs.remove(TARGET)
    #     exp1city(TRIAL, SOURCEs, TARGET)

    '''
    Experiment6:
    比較手法 KNN
    '''
    # for TARGET in CITIEs4:
    #     expKNN(TARGET)
    # for city in CITIEs4:
    #     analysisKNN(city)

    '''
    Experiment7:
    比較手法　LI
    '''
    # for TARGET in CITIEs4:
    #     expLI(TARGET)

    '''
    Experiment8:
    比較手法 FNN
    '''
    # for TARGET in CITIEs4:
    #     expFNN(TRIAL, TARGET)

    '''
    Experiment9:
    提案手法
    '''
    # CITIEs4 = ["BeiJing"]
    # for TARGET in CITIEs4:
    #     expProposal(TRIAL, TARGET)

    '''
    距離計算
    '''
    # c = pd.read_csv("rawdata/zheng2015/city.csv", index_col="name_english")
    # s = CITIES[0]
    # lat_local = c.at[s, "latitude"]
    # lon_local = c.at[s, "longitude"]
    # distance = list()
    # from source.utility import get_dist_angle
    # for t in CITIES:
    #     lat = c.at[t, "latitude"]
    #     lon = c.at[t, "longitude"]
    #     result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
    #     distance.append(str(result["distance"]))
    #
    # with open("tmp/distance.csv", "w") as outfile:
    #     outfile.write("{}\n".format(",".join(CITIES)))
    #     outfile.write("{}\n".format(",".join(distance)))

    '''
    MMD計算
    '''
    # alpha = 1
    # label = "1"
    # print("--- alpha = " + label + "---")
    #
    # print("* pre-computing is start")
    # for city in CITIEs20:
    #     mmd_pre = MMD_preComputed(city, alpha, 24 * 30 * 6)
    #     mmd_pre()
    #     del mmd_pre
    #
    # with open("result/mmd_{}.csv".format(label), "w") as outfile:
    #     outfile.write("target,{}\n".format(",".join(CITIEs20)))
    #
    # sourceDict = dict()
    # for SOURCE in CITIEs4:
    #     sourceDict[SOURCE] = dict(zip(CITIEs4, [0]*len(CITIEs4)))
    #
    # for SOURCE in CITIEs4:
    #
    #     print("* SOURCE = " + SOURCE)
    #     with open("result/mmd_{}.csv".format(label), "a") as outfile:
    #         outfile.write(SOURCE)
    #
    #     for TARGET in CITIEs20:
    #         print("\t * TARGET = " + TARGET)
    #
    #         if TARGET == SOURCE:
    #             result = 0.0
    #
    #         elif TARGET in CITIEs4:
    #             if sourceDict[SOURCE][TARGET] != 0:
    #                 result = sourceDict[SOURCE][TARGET]
    #             else:
    #                 mmd = MMD(SOURCE, TARGET, alpha)
    #                 result = mmd()
    #                 result = float(result) * float(result)
    #                 sourceDict[SOURCE][TARGET] = result
    #                 sourceDict[TARGET][SOURCE] = result
    #         else:
    #             mmd = MMD(SOURCE, TARGET, alpha)
    #             result = mmd()
    #             result = float(result) * float(result)
    #
    #         with open("result/mmd_{}.csv".format(label), "a") as outfile:
    #             outfile.write(",{}".format(str(result)))
    #
    #     with open("result/mmd_{}.csv".format(label), "a") as outfile:
    #         outfile.write("\n")