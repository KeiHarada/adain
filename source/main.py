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
from source.func import makeDataset_single
from source.func import makeDataset_multi
from source.func import makeDataset_mmd
from source.utility import get_dist_angle
from source.utility import MMD
from source.utility import MMD_preComputed
from source.utility import memory_limit
from source.func import objective
from source.func import evaluate
from source.func import re_evaluate

def experiment0(LOOP, TRIAL, ATTRIBUTE, CITY, TRAIN_RATE, VALID_RATE):

    # input dimension
    dataDim = pickle.load(open("dataset/dataDim.pickle", "rb"))
    inputDim_local_static = dataDim["static"] # static attribute
    inputDim_local_seq = dataDim["sequence"]  # sequence attribute
    inputDim_others_static = dataDim["static"] + 2  # static attribute + distance + angle
    inputDim_others_seq = dataDim["sequence"] + 1  # sequence attribute + an aqi value

    # save input dimension
    with open("model/inputDim.pickle", "wb") as fl:
        dc = {"local_static": inputDim_local_static, "local_seq": inputDim_local_seq,
              "others_static": inputDim_others_static, "others_seq": inputDim_others_seq}
        pickle.dump(dc, fl)

    # to devide the dataset
    station_all = pickle.load(open("dataset/stationAll.pickle", "rb"))
    TRAIN_NUM = int(len(station_all) * TRAIN_RATE)
    VALID_NUM = int(TRAIN_NUM * VALID_RATE)
    if VALID_NUM < 1:
        VALID_NUM = 1
    TEST_NUM = len(station_all)-(TRAIN_NUM+VALID_NUM)

    # statictics of dataset
    print("# of train = "+str(TRAIN_NUM))
    print("# of valid = "+str(VALID_NUM))
    print("# of test = "+str(TEST_NUM))

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(LOOP):
        print("---LOOP "+str(loop).zfill(2)+"---")
        start = time.time()

        # train, valid, test sets
        stations = station_all.copy()
        random.shuffle(stations)
        train = stations[:TRAIN_NUM]
        valid = stations[TRAIN_NUM:TRAIN_NUM+VALID_NUM]
        test = stations[TRAIN_NUM+VALID_NUM:TRAIN_NUM+VALID_NUM+TEST_NUM]

        # temporally save train, valid, test sets
        with open("tmp/trainset.pickle", "wb") as pl:
            pickle.dump(train, pl)
        with open("tmp/validset.pickle", "wb") as pl:
            pickle.dump(valid, pl)
        with open("tmp/testset.pickle", "wb") as pl:
            pickle.dump(test, pl)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        print("* training ... ")
        study = optuna.create_study()
        study.optimize(objective, n_trials=TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/" + str(study.best_trial.number).zfill(4) + "_model.pickle")
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "model", CITY)
        with open(path, "wb") as pl:
            pickle.dump(model_state_dict, pl)

        # save dataset
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "train", CITY)
        with open(path, "wb") as pl:
            pickle.dump(train, pl)
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "valid", CITY)
        with open(path, "wb") as pl:
            pickle.dump(valid, pl)
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "test", CITY)
        with open(path, "wb") as pl:
            pickle.dump(test, pl)

        # save train log
        with open("tmp/" + str(study.best_trial.number).zfill(4) + "_log.pickle", "rb") as pl:
            log = pickle.load(pl)
            path = "log/{}_{}_{}_{}.csv".format(ATTRIBUTE, str(loop).zfill(2), "log", CITY)
            log.to_csv(path, index=False)

        # load best model
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "model", CITY)
        model_state_dict = pickle.load(open(path, "rb"))

        # evaluate
        print("* evaluating ... ")
        rmse, accuracy = evaluate(model_state_dict, train, test)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        t = (time.time() - start) / (60*60)
        print("time = "+str(t)+" [hours]")
        print("---LOOP "+str(loop).zfill(2)+"---")

    path = "result/result_{}_{}.csv".format(ATTRIBUTE, CITY)
    with open(path, "w") as result:
        result.write("--------------------------------------------\n" +
                     "CITY = " + CITY + "\n" +
                     "MODEL_ATTRIBUTE = " + ATTRIBUTE + "\n" +
                     "TRAIN_NUM = " + str(TRAIN_NUM) + "\n" +
                     "VARID_NUM = " + str(VALID_NUM) + "\n" +
                     "TEST_NUM = " + str(TEST_NUM) + "\n" +
                     "--------------------------------------------\n")

    # to output
    rmse = str(np.average(np.array(rmse_list)))
    rmse_list = list(map(lambda x: str(x), rmse_list))
    accuracy = str(np.average(np.array(accuracy_list)))
    accuracy_list = list(map(lambda x: str(x), accuracy_list))

    with open(path, "a") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("model,{}\n".format(CITY))
        for i in range(len(rmse_list)):
            result.write("{},{}\n".format(str(i).zfill(2), rmse_list[i]))
        result.write("average,{}\n".format(rmse))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("model,{}\n".format(CITY))
        for i in range(len(accuracy_list)):
            result.write("{},{}\n".format(str(i).zfill(2), accuracy_list[i]))
        result.write("average,{}\n".format(accuracy))

def experiment1(LOOP, TRIAL, ATTRIBUTE, SOURCE, TARGETs):

    '''
    Train: Source city
    Test: Source, city, Target city
    '''

    # input dimension
    dataDim = pickle.load(open("dataset/dataDim.pickle", "rb"))
    inputDim_local_static = dataDim["static"] # static attribute
    inputDim_local_seq = dataDim["sequence"]  # sequence attribute
    inputDim_others_static = dataDim["static"] + 2  # static attribute + distance + angle
    inputDim_others_seq = dataDim["sequence"] + 1  # sequence attribute + an aqi value

    # save input dimension
    with open("model/inputDim.pickle", "wb") as fl:
        dc = {"local_static": inputDim_local_static, "local_seq": inputDim_local_seq,
              "others_static": inputDim_others_static, "others_seq": inputDim_others_seq}
        pickle.dump(dc, fl)

    # load source and target stations
    station_source = pickle.load(open("dataset/station_"+SOURCE+".pickle", "rb"))
    station_target = []
    for i in range(len(TARGETs)):
        station_target.append(pickle.load(open("dataset/station_"+TARGETs[i]+".pickle", "rb")))

    # constant value: the number of train, validate, test datasets
    TRAIN_NUM = 5
    VALID_NUM = 1
    TEST_NUM = 5

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    # statictics of dataset
    print("# of train = "+str(TRAIN_NUM))
    print("# of valid = "+str(VALID_NUM))
    print("# of test = "+str(TEST_NUM))

    for loop in range(LOOP):

        # to evaluate
        rmse_tmp = list()
        accuracy_tmp = list()

        print("---LOOP " + str(loop).zfill(2) + "---")
        start = time.time()

        # shuffle for LOOP
        source = station_source.copy()
        random.shuffle(source)
        target = []
        for i in range(len(TARGETs)):
            target.append(station_target[i].copy())
            random.shuffle(target[i])

        # select train, validate, test sets
        train = source[:TRAIN_NUM]
        valid = source[TRAIN_NUM:TRAIN_NUM+VALID_NUM]
        test_source = source[TRAIN_NUM+VALID_NUM:TRAIN_NUM+VALID_NUM+TEST_NUM]
        test_target = []
        for i in range(len(target)):
            test_target.append(target[i][:TEST_NUM])

        # saving train, valid, test sets
        with open("tmp/trainset.pickle", "wb") as pl:
            pickle.dump(train, pl)
        with open("tmp/validset.pickle", "wb") as pl:
            pickle.dump(valid, pl)
        with open("tmp/testset_"+SOURCE+".pickle", "wb") as pl:
            pickle.dump(test_source, pl)
        for i in range(len(test_target)):
            with open("tmp/testset_"+TARGETs[i]+".pickle", "wb") as pl:
                pickle.dump(test_target[i], pl)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        print("train in "+SOURCE)
        study = optuna.create_study()
        study.optimize(objective, n_trials=TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/" + str(study.best_trial.number).zfill(4) + "_model.pickle")
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "model", SOURCE)
        with open(path, "wb") as pl:
            pickle.dump(model_state_dict, pl)

        # save dataset
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "train", SOURCE)
        with open(path, "wb") as pl:
            pickle.dump(train, pl)
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "valid", SOURCE)
        with open(path, "wb") as pl:
            pickle.dump(valid, pl)
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "test", SOURCE)
        with open(path, "wb") as pl:
            pickle.dump(test_source, pl)
        for i in range(len(test_target)):
            path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "test", TARGETs[i])
            with open(path, "wb") as pl:
                pickle.dump(test_target[i], pl)

        # save train log
        with open("tmp/" + str(study.best_trial.number).zfill(4) + "_log.pickle", "rb") as pl:
            log = pickle.load(pl)
            path = "log/{}_{}_{}_{}.csv".format(ATTRIBUTE, str(loop).zfill(2), "log", SOURCE)
            log.to_csv(path, index=False)

        # load best model
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "model", SOURCE)
        model_state_dict = pickle.load(open(path, "rb"))

        # evaluate
        print("* evaluate in "+SOURCE)
        rmse, accuracy = evaluate(model_state_dict, train, test_source)
        rmse_tmp.append(rmse)
        accuracy_tmp.append(accuracy)

        for i in range(len(test_target)):
            print("* evaluate in "+TARGETs[i])
            rmse, accuracy = evaluate(model_state_dict, train, test_target[i])
            rmse_tmp.append(rmse)
            accuracy_tmp.append(accuracy)

        rmse_list.append(rmse_tmp)
        accuracy_list.append(accuracy_tmp)

        # time point
        t = (time.time() - start) / (60 * 60)
        print("time = " + str(t) + " [hours]")
        print("---LOOP " + str(loop).zfill(2) + "---")

    path = "result/result_{}_{}.csv".format(ATTRIBUTE, SOURCE)
    with open(path, "w") as result:
        result.write("--------------------------------------------\n" +
                     "SOURCE_CITY = " + SOURCE + "\n" +
                     "TARGET_CITY = " + ",".join(TARGETs) + "\n" +
                     "MODEL_ATTRIBUTE = " + ATTRIBUTE + "\n" +
                     "TRAIN_NUM = " + str(TRAIN_NUM) + "\n" +
                     "VAlID_NUM = " + str(VALID_NUM) + "\n" +
                     "TEST_NUM = " + str(TEST_NUM) + "\n" +
                     "--------------------------------------------\n")
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
    for target in TARGETs:
        lat = c.at[target, "latitude"]
        lon = c.at[target, "longitude"]
        result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
        distance.append(str(result["distance"]/1000.0))

    with open(path, "a") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("model,{},{}\n".format(SOURCE, ",".join(TARGETs)))
        for i in range(len(rmse_list)):
            result.write("{},{}\n".format(str(i).zfill(2), ",".join(rmse_list[i])))
        result.write("average,{}\n".format(",".join(rmse)))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("model,{},{}\n".format(SOURCE, ",".join(TARGETs)))
        for i in range(len(accuracy_list)):
            result.write("{},{}\n".format(str(i).zfill(2), ",".join(accuracy_list[i])))
        result.write("average,{}\n".format(",".join(accuracy)))
        result.write("--------------------------------------------\n")
        result.write("Distance\n")
        result.write("----------\n")
        result.write("distance,0.0,{}\n".format(",".join(distance)))

def experiment2(TRIAL, ATTRIBUTE, SOURCE, TARGETs, VALID_RATE):

    '''
    Train: Source city
    Test: Source, city, Target city
    '''

    # input dimension
    dataDim = pickle.load(open("dataset/dataDim.pickle", "rb"))
    inputDim_local_static = dataDim["static"] # static attribute
    inputDim_local_seq = dataDim["sequence"]  # sequence attribute
    inputDim_others_static = dataDim["static"] + 2  # static attribute + distance + angle
    inputDim_others_seq = dataDim["sequence"] + 1  # sequence attribute + an aqi value

    # save input dimension
    with open("model/inputDim.pickle", "wb") as fl:
        dc = {"local_static": inputDim_local_static, "local_seq": inputDim_local_seq,
              "others_static": inputDim_others_static, "others_seq": inputDim_others_seq}
        pickle.dump(dc, fl)

    # load source and target stations
    station_source = pickle.load(open("dataset/station_"+SOURCE+".pickle", "rb"))
    station_target = []
    for i in range(len(TARGETs)):
        station_target.append(pickle.load(open("dataset/station_"+TARGETs[i]+".pickle", "rb")))

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    # statictics of dataset
    print("# of train = MAX")
    print("# of test = MAX")


    # to evaluate
    rmse_tmp = list()
    accuracy_tmp = list()
    loop = 0

    start = time.time()

    # shuffle
    source = station_source.copy()
    random.shuffle(source)
    target = []
    for i in range(len(TARGETs)):
        target.append(station_target[i].copy())
        random.shuffle(target[i])

    # select train, validate, test sets
    VALID_NUM = int(len(source) * VALID_RATE)
    train = source[VALID_NUM:]
    valid = source[:VALID_NUM]
    test_target = []
    for i in range(len(target)):
        test_target.append(target[i])

    # saving train, valid, test sets
    with open("tmp/trainset.pickle", "wb") as pl:
        pickle.dump(train, pl)
    with open("tmp/validset.pickle", "wb") as pl:
        pickle.dump(valid, pl)
    for i in range(len(test_target)):
        with open("tmp/testset_"+TARGETs[i]+".pickle", "wb") as pl:
            pickle.dump(test_target[i], pl)

    # training & parameter tuning by optuna
    # -- activate function, optimizer, eopchs, batch size
    print("train in "+SOURCE)
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL)

    # save best model
    model_state_dict = torch.load("tmp/" + str(study.best_trial.number).zfill(4) + "_model.pickle")
    path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "model", SOURCE)
    with open(path, "wb") as pl:
        pickle.dump(model_state_dict, pl)

    # save dataset
    path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "train", SOURCE)
    with open(path, "wb") as pl:
        pickle.dump(train, pl)
    path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "valid", SOURCE)
    with open(path, "wb") as pl:
        pickle.dump(valid, pl)
    for i in range(len(test_target)):
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "test", TARGETs[i])
        with open(path, "wb") as pl:
            pickle.dump(test_target[i], pl)

    # save train log
    with open("tmp/" + str(study.best_trial.number).zfill(4) + "_log.pickle", "rb") as pl:
        log = pickle.load(pl)
        path = "log/{}_{}_{}_{}.csv".format(ATTRIBUTE, str(loop).zfill(2), "log", SOURCE)
        log.to_csv(path, index=False)

    # load best model
    path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "model", SOURCE)
    model_state_dict = pickle.load(open(path, "rb"))

    # evaluate
    for i in range(len(test_target)):
        print("* evaluate in "+TARGETs[i])
        rmse, accuracy = evaluate(model_state_dict, train, test_target[i])
        rmse_tmp.append(rmse)
        accuracy_tmp.append(accuracy)

    rmse_list.append(rmse_tmp)
    accuracy_list.append(accuracy_tmp)

    # time
    t = (time.time() - start) / (60 * 60)
    print("time = " + str(t) + " [hours]")

    path = "result/result_{}_{}.csv".format(ATTRIBUTE, SOURCE)
    with open(path, "w") as result:
        result.write("--------------------------------------------\n" +
                     "SOURCE_CITY = " + SOURCE + "\n" +
                     "TARGET_CITY = " + ",".join(TARGETs) + "\n" +
                     "MODEL_ATTRIBUTE = " + ATTRIBUTE + "\n" +
                     "TRAIN_NUM = MAX\n" +
                     "TEST_NUM = MAX\n" +
                     "--------------------------------------------\n")
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
    for target in TARGETs:
        lat = c.at[target, "latitude"]
        lon = c.at[target, "longitude"]
        result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
        distance.append(str(result["distance"]/1000.0))

    with open(path, "a") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("model,{},{}\n".format(SOURCE, ",".join(TARGETs)))
        for i in range(len(rmse_list)):
            result.write("{},NULL,{}\n".format(str(i).zfill(2), ",".join(rmse_list[i])))
        result.write("average,NULL,{}\n".format(",".join(rmse)))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("model,{},{}\n".format(SOURCE, ",".join(TARGETs)))
        for i in range(len(accuracy_list)):
            result.write("{},NULL,{}\n".format(str(i).zfill(2), ",".join(accuracy_list[i])))
        result.write("average,NULL,{}\n".format(",".join(accuracy)))
        result.write("--------------------------------------------\n")
        result.write("Distance\n")
        result.write("----------\n")
        result.write("distance,0.0,{}\n".format(",".join(distance)))

def experiment3(LOOP, TRIAL, ATTRIBUTE, SOURCEs, TARGET):

    '''
    Train: CITIEs
    Test: a CITY
    '''

    # input dimension
    dataDim = pickle.load(open("dataset/dataDim.pickle", "rb"))
    inputDim_local_static = dataDim["static"] # static attribute
    inputDim_local_seq = dataDim["sequence"]  # sequence attribute
    inputDim_others_static = dataDim["static"] + 2  # static attribute + distance + angle
    inputDim_others_seq = dataDim["sequence"] + 1  # sequence attribute + an aqi value

    # save input dimension
    with open("model/inputDim.pickle", "wb") as fl:
        dc = {"local_static": inputDim_local_static, "local_seq": inputDim_local_seq,
              "others_static": inputDim_others_static, "others_seq": inputDim_others_seq}
        pickle.dump(dc, fl)

    # load source and target stations
    station_source = []
    for i in range(len(SOURCEs)):
        station_source.append(pickle.load(open("dataset/station_"+SOURCEs[i]+".pickle", "rb")))

    station_target = pickle.load(open("dataset/station_"+TARGET+".pickle", "rb"))

    # constant value: the number of train, validate, test datasets
    TRAIN_NUM = 5
    VALID_NUM = 1
    TEST_NUM = 5

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    # statictics of dataset
    print("# of train = "+str(TRAIN_NUM))
    print("# of valid = "+str(VALID_NUM))
    print("# of test = "+str(TEST_NUM))

    for loop in range(LOOP):

        # to evaluate
        rmse_tmp = list()
        accuracy_tmp = list()

        print("---LOOP " + str(loop).zfill(2) + "---")
        start = time.time()

        # shuffle for LOOP
        source = []
        for i in range(len(SOURCEs)):
            source.append(station_source[i].copy())
            random.shuffle(source[i])

        target = station_target.copy()
        random.shuffle(target)

        # select train, validate, test sets
        train = []
        for i in range(len(source)):
            train += source[i][:TRAIN_NUM]

        valid = target[TRAIN_NUM:TRAIN_NUM+VALID_NUM]
        test = target[TRAIN_NUM+VALID_NUM:TRAIN_NUM+VALID_NUM+TEST_NUM]

        # saving train, valid, test sets
        with open("tmp/trainset.pickle", "wb") as pl:
            pickle.dump(train, pl)
        with open("tmp/validset.pickle", "wb") as pl:
            pickle.dump(valid, pl)
        with open("tmp/testset.pickle", "wb") as pl:
            pickle.dump(test, pl)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        print("train")
        study = optuna.create_study()
        study.optimize(objective, n_trials=TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/" + str(study.best_trial.number).zfill(4) + "_model.pickle")
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "model", TARGET)
        with open(path, "wb") as pl:
            pickle.dump(model_state_dict, pl)

        # save dataset
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "train", TARGET)
        with open(path, "wb") as pl:
            pickle.dump(train, pl)
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "valid", TARGET)
        with open(path, "wb") as pl:
            pickle.dump(valid, pl)
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "test", TARGET)
        with open(path, "wb") as pl:
            pickle.dump(test, pl)

        # save train log
        with open("tmp/" + str(study.best_trial.number).zfill(4) + "_log.pickle", "rb") as pl:
            log = pickle.load(pl)
            path = "log/{}_{}_{}_{}.csv".format(ATTRIBUTE, str(loop).zfill(2), "log", TARGET)
            log.to_csv(path, index=False)

        # load best model
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "model", TARGET)
        model_state_dict = pickle.load(open(path, "rb"))

        # evaluate
        print("* evaluate")
        rmse, accuracy = evaluate(model_state_dict, train, test)
        rmse_tmp.append(rmse)
        accuracy_tmp.append(accuracy)

        rmse_list.append(rmse_tmp)
        accuracy_list.append(accuracy_tmp)

        # time point
        t = (time.time() - start) / (60 * 60)
        print("time = " + str(t) + " [hours]")
        print("---LOOP " + str(loop).zfill(2) + "---")

    path = "result/result_{}_{}.csv".format(ATTRIBUTE, TARGET)
    with open(path, "w") as result:
        result.write("--------------------------------------------\n" +
                     "TARGET_CITY = " + TARGET + "\n" +
                     "MODEL_ATTRIBUTE = " + ATTRIBUTE + "\n" +
                     "TRAIN_NUM = " + str(TRAIN_NUM) + "\n" +
                     "VAlID_NUM = " + str(VALID_NUM) + "\n" +
                     "TEST_NUM = " + str(TEST_NUM) + "\n" +
                     "--------------------------------------------\n")
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

    with open(path, "a") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("model,{}\n".format(TARGET))
        for i in range(len(rmse_list)):
            result.write("{},{}\n".format(str(i).zfill(2), ",".join(rmse_list[i])))
        result.write("average,{}\n".format(",".join(rmse)))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("model,{}\n".format(TARGET))
        for i in range(len(accuracy_list)):
            result.write("{},{}\n".format(str(i).zfill(2), ",".join(accuracy_list[i])))
        result.write("average,{}\n".format(",".join(accuracy)))

def analysis(source, targets):
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from source.utility import calc_correct
    from source.utility import data_interpolate
    import matplotlib.pyplot as plt

    x = [0, 51, 101, 151, 201, 301, 501, 1001]

    aqi_attribute = ["pm25", "pm10", "no2", "co", "o3", "so2"]
    model_attribute = "pm25"
    dtype = {att: "float" for att in aqi_attribute}
    dtype["sid"], dtype["time"] = "object", "object"

    print("////// "+source+" //////")
    aqi_source = pd.read_csv("database/aqi/aqi_" + source + ".csv", dtype=dtype)
    df = data_interpolate(aqi_source[[model_attribute]])
    aqi_source = pd.concat([aqi_source.drop(aqi_attribute, axis=1), df], axis=1)
    # print("# of stations = {}".format(len(set(aqi_source["sid"]))))
    tmp = list()
    for i in range(len(x)-1):
        aqi = aqi_source[aqi_source[model_attribute] >= float(x[i])]
        aqi = aqi[aqi[model_attribute] < float(x[i+1])]
        tmp.append(len(aqi))
    # print(tmp)
    tmp = list(map(lambda a: a/sum(tmp), tmp))
    # print(tmp)
    # print(aqi_source.describe())
    #pd.cut(aqi_source[model_attribute], x, right=False).value_counts().sort_index().plot.bar(color='gray')
    #plt.savefig("tmp/"+SOURCE+".pdf")

    aqi = aqi_source[model_attribute].values
    mean = np.mean(aqi)
    print("////// "+source+" //////")

    rmse_s = list()
    accuracy_s = list()

    for target in targets:

        rmse_t = list()
        accuracy_t = list()

        print("////// "+target+" //////")
        aqi_target = pd.read_csv("database/aqi/aqi_" + target + ".csv", dtype=dtype)
        df = data_interpolate(aqi_target[[model_attribute]])
        aqi_target = pd.concat([aqi_target.drop(aqi_attribute, axis=1), df], axis=1)
        # print("# of stations = {}".format(len(set(aqi_target["sid"]))))
        tmp = list()
        for i in range(len(x) - 1):
            aqi = aqi_target[aqi_target[model_attribute] >= float(x[i])]
            aqi = aqi[aqi[model_attribute] < float(x[i + 1])]
            tmp.append(len(aqi))
        # print(tmp)
        tmp = list(map(lambda a: a / sum(tmp), tmp))
        # print(tmp)
        # print(aqi_target.describe())
        # pd.cut(aqi_target[model_attribute], x, right=False).value_counts().sort_index().plot.bar(color='gray')
        # plt.savefig("tmp/" + target + ".pdf")
        # print("----------------------")
        for i in range(5):

            # train
            train = pickle.load(open("model/"+source + "2" + target + "_" + model_attribute + "_"
                                     +str(i).zfill(2)+"_trainset.pickle", "rb"))
            # test: source
            test_s = pickle.load(open("model/"+source+"2"+target+"_"+model_attribute+"_"
                                     +str(i).zfill(2)+"_testset_source.pickle", "rb"))
            # test: target
            test_t = pickle.load(open("dataset/station_"+target+".pickle", "rb"))
            random.shuffle(test_t)
            test_t = test_t[:5]

            # aqi = aqi_source[aqi_source["sid"].isin(train)]
            # aqi = aqi[model_attribute].values
            # seikai = np.full(len(aqi), mean)
            # print(np.sqrt(mean_squared_error(aqi, seikai)))
            # print(calc_correct(aqi, seikai) / len(aqi))
            # print(train)
            # print(aqi.describe())
            # pd.cut(aqi[model_attribute], x, right=False).value_counts().sort_index().plot.bar(color='gray')

            aqi = aqi_source[aqi_source["sid"].isin(test_s)]
            aqi = aqi[model_attribute].values
            seikai = np.full(len(aqi), mean)
            rmse_s.append(np.sqrt(mean_squared_error(aqi, seikai)))
            accuracy_s.append(calc_correct(aqi, seikai) / len(aqi))
            print(np.sqrt(mean_squared_error(aqi, seikai)))
            print(calc_correct(aqi, seikai) / len(aqi))
            print("")
            # print(test_s)
            # print(aqi.describe())
            # pd.cut(aqi[model_attribute], x, right=False).value_counts().sort_index().plot.bar(color='gray')

            aqi = aqi_target[aqi_target["sid"].isin(test_t)]
            aqi = aqi[model_attribute].values
            seikai = np.full(len(aqi), mean)
            rmse_t.append(np.sqrt(mean_squared_error(aqi, seikai)))
            accuracy_t.append(calc_correct(aqi, seikai) / len(aqi))
            print(np.sqrt(mean_squared_error(aqi, seikai)))
            print(calc_correct(aqi, seikai) / len(aqi))
            # print(test_t)
            # print(aqi.describe())
            # pd.cut(aqi[model_attribute], x, right=False).value_counts().sort_index().plot.bar(color='gray')

            plt.show()
            print("----------------------")
        print(np.mean(np.array(rmse_s)))
        print(np.mean(np.array(accuracy_s)))
        print("")
        print(np.mean(np.array(rmse_t)))
        print(np.mean(np.array(accuracy_t)))
        print("////// "+target+" //////")

def reEvaluate(LOOP, ATTRIBUTE, SOURCE, TARGETs):

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(LOOP):

        start = time.time()

        # to evaluate
        rmse_tmp = list()
        accuracy_tmp = list()

        # load train, test dataset
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "train", SOURCE)
        train = pickle.load(open(path, "rb"))
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "test", SOURCE)
        test_source = pickle.load(open(path, "rb"))
        test_target = []
        for i in range(len(TARGETs)):
            path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "test", TARGETs[i])
            test_target.append(pickle.load(open(path, "rb")))

        # load model
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(loop).zfill(2), "model", SOURCE)
        model_state_dict = pickle.load(open(path, "rb"))

        # evaluate
        print("* evaluate in " + SOURCE)
        rmse, accuracy = re_evaluate(model_state_dict, train, test_source, loop, SOURCE)
        rmse_tmp.append(rmse)
        accuracy_tmp.append(accuracy)

        for i in range(len(test_target)):
            print("* evaluate in " + TARGETs[i])
            rmse, accuracy = re_evaluate(model_state_dict, train, test_target[i], loop, TARGETs[i])
            rmse_tmp.append(rmse)
            accuracy_tmp.append(accuracy)

        rmse_list.append(rmse_tmp)
        accuracy_list.append(accuracy_tmp)

        # time point
        t = (time.time() - start) / (60 * 60)
        print("time = " + str(t) + " [hours]")
        print("---LOOP " + str(loop).zfill(2) + "---")

    # to output
    path = "result/re_result_{}_{}.csv".format(ATTRIBUTE, SOURCE)
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
    for target in TARGETs:
        lat = c.at[target, "latitude"]
        lon = c.at[target, "longitude"]
        result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
        distance.append(str(result["distance"]/1000.0))

    with open(path, "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("model,{},{}\n".format(SOURCE, ",".join(TARGETs)))
        for i in range(len(rmse_list)):
            result.write("{},{}\n".format(str(i).zfill(2), ",".join(rmse_list[i])))
        result.write("average,{}\n".format(",".join(rmse)))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("model,{},{}\n".format(SOURCE, ",".join(TARGETs)))
        for i in range(len(accuracy_list)):
            result.write("{},{}\n".format(str(i).zfill(2), ",".join(accuracy_list[i])))
        result.write("average,{}\n".format(",".join(accuracy)))
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
    TRAIN_RATE = 0.67
    VALID_RATE = 0.1
    LSTM_DATA_WIDTH = 24
    LOOP = 3
    TRIAL = 1

    '''
    Experiment0:
    Re-experiment of AAAI'18
    '''
    #SOURCE = "BeiJing"
    #TARGET = "TianJin"
    #makeDataset_single(SOURCE, ATTRIBUTE, LSTM_DATA_WIDTH, 24*30)
    #experiment0(LOOP, TRIAL, ATTRIBUTE, SOURCE, TRAIN_RATE, VALID_RATE)

    '''
    Experiment1:
    仮説検証のためのone-to-oneの実験
    '''
    # # test=5stationsある都市を選択
    # # 気象データが全部Nullの都市は無視
    # CITIEs = list()
    # for city in list(pd.read_csv("rawdata/zheng2015/city.csv")["name_english"]):
    #     with open("database/station/station_"+city+".csv", "r") as infile:
    #         infile = infile.readlines()[1:] # 1行目を無視
    #         if len(infile) >= 5:
    #             CITIEs.append(city)
    # CITIEs.remove("JiNan")
    # CITIEs.remove("HeYuan")
    # CITIEs.remove("JieYang")
    # CITIEs.remove("ShaoGuan")
    # CITIEs.remove("DaTong")
    # CITIEs.remove("DeZhou")
    # CITIEs.remove("BinZhou")
    # CITIEs.remove("DongYing")
    # CITIEs.remove("ChenZhou")
    #
    # # make dataset
    # makeDataset_multi(CITIEs, ATTRIBUTE, LSTM_DATA_WIDTH, 24 * 30 * 3)
    #
    # # Cluster 1: BeiJing[1], TianJin[1.5], ShiJiaZhuang[2]
    # # Cluster 2: ShenZhen[1], GuangZhou[1], ChaoZhou[3]
    # SOURCEs = ["BeiJing", "TianJin", "ShiJiaZhuang", "ShenZhen", "GuangZhou", "CangZhou"]
    # for SOURCE in SOURCEs:
    #     TARGETs = CITIEs.copy()
    #     TARGETs.remove(SOURCE)
    #     experiment1(LOOP, TRIAL, ATTRIBUTE, SOURCE, TARGETs)

    '''
    Experiment2:
    最大性能を検証するための実験
    '''
    # # test=5stationsある都市を選択
    # # 気象データが全部Nullの都市は無視
    # CITIEs = list()
    # for city in list(pd.read_csv("rawdata/zheng2015/city.csv")["name_english"]):
    #     with open("database/station/station_"+city+".csv", "r") as infile:
    #         infile = infile.readlines()[1:] # 1行目を無視
    #         if len(infile) >= 5:
    #             CITIEs.append(city)
    # CITIEs.remove("JiNan")
    # CITIEs.remove("HeYuan")
    # CITIEs.remove("JieYang")
    # CITIEs.remove("ShaoGuan")
    # CITIEs.remove("DaTong")
    # CITIEs.remove("DeZhou")
    # CITIEs.remove("BinZhou")
    # CITIEs.remove("DongYing")
    # CITIEs.remove("ChenZhou")
    #
    # # make dataset
    # makeDataset_multi(CITIEs, ATTRIBUTE, LSTM_DATA_WIDTH, 24 * 30 * 3)
    #
    # # Cluster 1: BeiJing[1], TianJin[1.5], ShiJiaZhuang[2]
    # # Cluster 2: ShenZhen[1], GuangZhou[1], ChaoZhou[3]
    # SOURCEs = ["BeiJing", "TianJin", "ShiJiaZhuang", "ShenZhen", "GuangZhou", "CangZhou"]
    # for SOURCE in SOURCEs:
    #     TARGETs = CITIEs.copy()
    #     TARGETs.remove(SOURCE)
    #     experiment2(TRIAL, ATTRIBUTE, SOURCE, TARGETs, VALID_RATE)

    '''
    Experiment3:
    マルチソースで実験
    '''
    # # test=5stationsある都市を選択
    # # 気象データが全部Nullの都市は無視3
    CITIEs = list()
    for city in list(pd.read_csv("rawdata/zheng2015/city.csv")["name_english"]):
        with open("database/station/station_"+city+".csv", "r") as infile:
            infile = infile.readlines()[1:] # 1行目を無視
            if len(infile) >= 5:
                CITIEs.append(city)
    CITIEs.remove("JiNan")
    CITIEs.remove("HeYuan")
    CITIEs.remove("JieYang")
    CITIEs.remove("ShaoGuan")
    CITIEs.remove("DaTong")
    CITIEs.remove("DeZhou")
    CITIEs.remove("BinZhou")
    CITIEs.remove("DongYing")
    CITIEs.remove("ChenZhou")

    # make dataset
    #makeDataset_multi(CITIEs, ATTRIBUTE, LSTM_DATA_WIDTH, 24 * 30 * 6)

    # Cluster 1: BeiJing[1], TianJin[1.5], ShiJiaZhuang[2]
    # Cluster 2: ShenZhen[1], GuangZhou[1], ChaoZhou[3]
    # TARGETs = ["BeiJing", "TianJin", "ShiJiaZhuang", "ShenZhen", "GuangZhou", "CangZhou"]
    TARGETs = ["TianJin", "ShenZhen", "GuangZhou"]
    for TARGET in TARGETs:
        experiment3(LOOP, TRIAL, ATTRIBUTE, CITIEs, TARGET)

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
    # # test=5stationsある都市を選択
    # # 気象データが全部Nullの都市は無視
    # #memory_limit()
    # CITIEs = list()
    # for city in list(pd.read_csv("rawdata/zheng2015/city.csv")["name_english"]):
    #     with open("database/station/station_"+city+".csv", "r") as infile:
    #         infile = infile.readlines()[1:] # 1行目を無視
    #         if len(infile) >= 5:
    #             CITIEs.append(city)
    # CITIEs.remove("JiNan")
    # CITIEs.remove("HeYuan")
    # CITIEs.remove("JieYang")
    # CITIEs.remove("ShaoGuan")
    # CITIEs.remove("DaTong")
    # CITIEs.remove("DeZhou")
    # CITIEs.remove("BinZhou")
    # CITIEs.remove("DongYing")
    # CITIEs.remove("ChenZhou")
    #
    # # Cluster 1: BeiJing[1], TianJin[1.5], ShiJiaZhuang[2]
    # # Cluster 2: ShenZhen[1], GuangZhou[1], ChaoZhou[3]
    # SOURCEs = ["BeiJing", "TianJin", "ShiJiaZhuang", "ShenZhen", "GuangZhou", "CangZhou"]
    # for alpha in [0.1, 1.0, 10.0]:
    #
    #     if alpha == 0.1:
    #         label = "01"
    #     elif alpha == 1.0:
    #         label = "1"
    #     else:
    #         label = "10"
    #
    #     print("--- alpha = " + label + "---")
    #
    #     print("* pre-computing is start")
    #
    #     for city in CITIEs:
    #         mmd_pre = MMD_preComputed(city, alpha, 24 * 30 * 6)
    #         mmd_pre()
    #         del mmd_pre
    #
    #     with open("result/result_mmd_"+label+".csv", "w") as outfile:
    #         outfile.write("target,{}\n".format(",".join(CITIEs)))
    #         for SOURCE in SOURCEs:
    #             print("* SOURCE = " + SOURCE)
    #             outfile.write(SOURCE)
    #             for TARGET in CITIEs:
    #                 print("\t * TARGET = " + TARGET)
    #                 mmd = MMD(SOURCE, TARGET, alpha)
    #                 result = mmd()
    #                 result = float(result) * float(result)
    #                 outfile.write(",{}".format(str(result)))
    #             outfile.write("\n")

    '''
    再実験
    '''
    #reEvaluate(LOOP, ATTRIBUTE, SOURCE, TARGETs)