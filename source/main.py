# to run on server
import sys
sys.path.append("/home/harada/Documents/WorkSpace/adain")
sys.path.append("/home")

import pickle
import time
import random
import torch
import optuna
import numpy as np
import pandas as pd
# from my library
from source.func import makeDataset
from source.func import makeTrainData
from source.func import makeTestData
from source.func import objective
from source.func import evaluate
from source.utility import get_dist_angle
from source.utility import MMD
from source.utility import MMD_preComputed


def expAAAI(TRIAL, CITY):

    '''
    AAAI 2018 の再実験
    '''

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):

        start = time.time()
        print("----------------")
        print("* SOURCE: {}{} ---".format(CITY, str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/AAAI18/train_{}{}/".format(CITY, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/AAAI18/test_{}{}/".format(CITY, str(loop)), fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective, n_trials=TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/AAAI18_{}{}.pkl".format(CITY, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/AAAI18_{}{}.csv".format(CITY, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/AAAI18_{}{}.pkl".format(CITY, str(loop)), "rb"))

        # evaluate
        print("* TARGET: {}{} ---".format(CITY, str(loop)))
        rmse, accuracy = evaluate(model_state_dict)
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
        print("*SOURCE: {}{} ---".format(SOURCE, str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Train/train_{}{}/".format(SOURCE, SOURCE, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Train/test_{}{}/".format(SOURCE, SOURCE, str(loop)), fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective, n_trials=TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/{}Train{}.pkl".format(SOURCE, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/{}Train{}.csv".format(SOURCE, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/{}Train{}.pkl".format(SOURCE, str(loop)), "rb"))

        # to evaluate
        rmse_tmp = list()
        accuracy_tmp = list()

        # evaluate
        print("*TARGET: {}{} ---".format(SOURCE, str(loop)))
        rmse, accuracy = evaluate(model_state_dict)
        rmse_tmp.append(rmse)
        accuracy_tmp.append(accuracy)

        for TARGET in TARGETs:

            # save dataset path
            with open("tmp/testPath.pkl", "wb") as fp:
                pickle.dump("dataset/{}Train/test_{}/".format(SOURCE, TARGET), fp)

            # evaluate
            print("*TARGET: {} ---".format(TARGET))
            rmse, accuracy = evaluate(model_state_dict)
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

    with open("result/{}Train.csv".format(SOURCE), "w") as result:
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
    print("*SOURCE: {} ---".format(SOURCE))

    # save dataset path
    with open("tmp/trainPath.pkl", "wb") as fp:
        pickle.dump("dataset/{}Train/train_{}Max/".format(SOURCE, SOURCE), fp)

    # training & parameter tuning by optuna
    # -- activate function, optimizer, eopchs, batch size
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL)

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
            pickle.dump("dataset/{}Train/test_{}/".format(SOURCE, TARGET), fp)

        # evaluate
        print("*TARGET: {} ---".format(TARGET))
        rmse, accuracy = evaluate(model_state_dict)
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
        print("* SOURCE: All{} ---".format(str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}19/train{}/".format(TARGET, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}19/test{}/".format(TARGET, str(loop)), fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective, n_trials=TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/{}19_{}.pkl".format(TARGET, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/{}19_{}.csv".format(TARGET, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/{}19_{}.pkl".format(TARGET, str(loop)), "rb"))

        # evaluate
        print("* TARGET: {}{} ---".format(TARGET, str(loop)))
        rmse, accuracy = evaluate(model_state_dict)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hours]".format(str((time.time() - start) / (60 * 60))))

    with open("result/{}19.csv".format(TARGET), "w") as result:
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
        print("* SOURCE: All{} ---".format(str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Dist5/train{}/".format(TARGET, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Dist5/test{}/".format(TARGET, str(loop)), fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective, n_trials=TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/{}Dist5_{}.pkl".format(TARGET, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/{}Dist5_{}.csv".format(TARGET, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/{}Dist5_{}.pkl".format(TARGET, str(loop)), "rb"))

        # evaluate
        print("* TARGET: {}{} ---".format(TARGET, str(loop)))
        rmse, accuracy = evaluate(model_state_dict)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hours]".format(str((time.time() - start) / (60 * 60))))

    with open("result/{}Dist5.csv".format(TARGET), "w") as result:
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

def exp1cities(TRIAL, SOURCEs, TARGET):

    '''
    他の都市で訓練したモデルでテスト
    '''

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):

        start = time.time()
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}1/test_{}/".format(TARGET, TARGET), fp)

        # to evaluate
        rmse_tmp = list()
        accuracy_tmp = list()

        for SOURCE in SOURCEs:

            print("*----------------")
            print("*SOURCE: {}{} ---".format(SOURCE, str(loop)))

            # save dataset path
            with open("tmp/trainPath.pkl", "wb") as fp:
                pickle.dump("dataset/{}1/train_{}{}/".format(TARGET, SOURCE, str(loop)), fp)

            # training & parameter tuning by optuna
            # -- activate function, optimizer, eopchs, batch size
            study = optuna.create_study()
            study.optimize(objective, n_trials=TRIAL)

            # save best model
            model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
            pickle.dump(model_state_dict, open("model/{}1_{}.pkl".format(TARGET, str(loop)), "wb"))

            # save train log
            log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
            log.to_csv("log/{}1_{}.csv".format(TARGET, str(loop)), index=False)

            # load best model
            model_state_dict = pickle.load(open("model/{}1_{}.pkl".format(TARGET, str(loop)), "rb"))

            # evaluate
            print("*TARGET: {}{} ---".format(TARGET, str(loop)))
            rmse, accuracy = evaluate(model_state_dict)
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

    with open("result/{}1.csv".format(TARGET), "w") as result:
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
    # 気象データが全部Nullの都市は無視
    CITIE20 = list()
    for city in list(pd.read_csv("rawdata/zheng2015/city.csv")["name_english"]):
        with open("database/station/station_"+city+".csv", "r") as infile:
            infile = infile.readlines()[1:] # 1行目を無視
            if len(infile) >= 5:
                CITIE20.append(city)
    CITIE20.remove("JiNan")
    CITIE20.remove("HeYuan")
    CITIE20.remove("JieYang")
    CITIE20.remove("ShaoGuan")
    CITIE20.remove("DaTong")
    CITIE20.remove("DeZhou")
    CITIE20.remove("BinZhou")
    CITIE20.remove("DongYing")
    CITIE20.remove("ChenZhou")

    '''
    4 cities
    '''
    CITIE4 = ["BeiJing", "TianJin", "ShenZhen", "GuangZhou"]

    '''
    create dataset
    '''
    makeDataset(CITIE20, ATTRIBUTE, LSTM_DATA_WIDTH, TIMEPERIOD)

    for TARGET in CITIE4:
        if TARGET == "BeiJing":
            SOURCEs = ["LangFang", "TianJin", "BaoDing", "TangShan", "ZhangJiaKou"]
        elif TARGET == "TianJin":
            SOURCEs = ["LangFang", "CangZhou", "TangShan", "BeiJing", "BaoDing"]
        elif TARGET == "ShenZhen":
            SOURCEs = ["XiangGang", "DongGuan", "HuiZhou", "JiangMen", "GuangZhou"]
        else:
            SOURCEs = ["FoShan", "DongGuan", "JiangMen", "ShenZhen", "HuiZhou"]

    '''
    Experiment0:
    AAAI'18 の再実験
    '''
    CITY = "BeiJing"
    expAAAI(TRIAL, CITY)

    '''
    Experiment1:
    物理距離と特徴距離の仮説検証実験
    '''
    for SOURCE in CITIE4:
        TARGETs = CITIE20.copy()
        TARGETs.remove(SOURCE)
        expDistance(TRIAL, SOURCE, TARGETs)

    '''
    Experiment2:
    最大性能を検証するための実験
    '''
    for SOURCE in CITIE4:
        TARGETs = CITIE20.copy()
        TARGETs.remove(SOURCE)
        expMAX(TRIAL, SOURCE, TARGETs)

    '''
    Experiment3:
    全都市で訓練したモデルの性能検証実験
    '''
    for TARGET in CITIE4:
        exp19cities(TRIAL, TARGET)

    '''
    Experiment4:
    距離の近い都市で訓練したモデルの性能検証実験
    '''
    for TARGET in CITIE4:
        exp5cities(TRIAL, TARGET)

    '''
    Experiment5:
    他都市で訓練したモデルでの性能検証実験
    '''
    for TARGET in CITIE4:
        SOURCEs = CITIE20.copy()
        SOURCEs.remove(TARGET)
        exp1cities(TRIAL, SOURCEs, TARGET)

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
    # for city in CITIE20:
    #     mmd_pre = MMD_preComputed(city, alpha, 24 * 30 * 6)
    #     mmd_pre()
    #     del mmd_pre
    #
    # with open("result/mmd_{}.csv".format(label), "w") as outfile:
    #     outfile.write("target,{}\n".format(",".join(CITIE20)))
    #
    # sourceDict = dict()
    # for SOURCE in CITIE4:
    #     sourceDict[SOURCE] = dict(zip(CITIE4, [0]*len(CITIE4)))
    #
    # for SOURCE in CITIE4:
    #
    #     print("* SOURCE = " + SOURCE)
    #     with open("result/mmd_{}.csv".format(label), "a") as outfile:
    #         outfile.write(SOURCE)
    #
    #     for TARGET in CITIE20:
    #         print("\t * TARGET = " + TARGET)
    #
    #         if TARGET == SOURCE:
    #             result = 0.0
    #
    #         elif TARGET in CITIE4:
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