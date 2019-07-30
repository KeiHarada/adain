import pickle
import time
import random
import torch
import optuna
import numpy as np
# from my library
from source.func import makeDataset0
from source.func import makeDataset1
from source.func import objective
from source.func import evaluate
from source.utility import Color

def experiment0(LOOP, TRIAL, ATTRIBUTE, CITY, TRAIN_RATE, VALID_RATE, LSTM_DATA_WIDTH):

    # input dimension
    dataDim = pickle.load(open("dataset/dataDim.pickle", "rb"))
    inputDim_static = dataDim["road"] + 2  # road attribute + distance + angle
    inputDim_seq_local = dataDim["meteorology"]  # meteorology attribute
    inputDim_seq_others = dataDim["meteorology"] + 1  # meteorology attribute + an aqi value

    # saving input dimension
    with open("model/inputDim.pickle", "wb") as fl:
        dc = {"static": inputDim_static,
              "seq_local": inputDim_seq_local,
              "seq_others": inputDim_seq_others}
        pickle.dump(dc, fl)

    # to devide the dataset
    station_all = pickle.load(open("dataset/stationAll.pickle", "rb"))
    TRAIN_NUM = int(len(station_all) * TRAIN_RATE)
    VARID_NUM = int(TRAIN_NUM * VALID_RATE)
    if VARID_NUM == 0:
        VARID_NUM = 1
    TEST_NUM = len(station_all)-(TRAIN_NUM+VARID_NUM)

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for i in range(LOOP):
        print("---LOOP "+str(i).zfill(2)+"---")
        start = time.time()

        stations = station_all.copy()
        random.shuffle(stations)
        train = stations[:TRAIN_NUM]
        valid = stations[TRAIN_NUM:TRAIN_NUM+VARID_NUM]
        test = stations[TRAIN_NUM+VARID_NUM:]

        # saving train, valid, test sets
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
        print(Color.GREEN + "OK" + Color.END)

        # evaluate
        print("* evaluating ... ")
        model, rmse, accuracy = evaluate(study.best_trial, train, test)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # saving model
        with open("model/" + CITY + "_" + ATTRIBUTE + "_" + str(i).zfill(2) + "_model.pickle", "wb") as pl:
            torch.save(model.state_dict(), pl)
        with open("model/" + CITY + "_" + ATTRIBUTE + "_" + str(i).zfill(2) + "_trainset.pickle", "wb") as pl:
            pickle.dump(train, pl)
        with open("model/" + CITY + "_" + ATTRIBUTE + "_" + str(i).zfill(2) + "_validset.pickle", "wb") as pl:
            pickle.dump(valid, pl)
        with open("model/" + CITY + "_" + ATTRIBUTE + "_" + str(i).zfill(2) + "_testset.pickle", "wb") as pl:
            pickle.dump(test, pl)

        t = (time.time() - start) / (60*60)
        print("time = "+str(t)+" [hours]")
        print("---LOOP "+str(i).zfill(2)+"---")

    # output results
    EPOCHS = 1
    BATCH_SIZE = 1
    LEARNING_RATE = 1

    with open("result/result_" + CITY + "_" + ATTRIBUTE + ".csv", "w") as result:
        result.write("--------------------------------------------\n" +
                     "CITY = " + CITY + "\n" +
                     "MODEL_ATTRIBUTE = " + ATTRIBUTE + "\n" +
                     "TRAIN_SIZE_RATE = " + str(TRAIN_RATE) + "\n" +
                     "VARID_SIZE_RATE = " + str(VALID_RATE) + "\n" +
                     "TRAIN_NUM = " + str(TRAIN_NUM) + "\n" +
                     "VARID_NUM = " + str(VARID_NUM) + "\n" +
                     "TEST_NUM = " + str(TEST_NUM) + "\n" +
                     "EPOCHS_NUM = " + str(EPOCHS) + "\n" +
                     "BATCH_SIZE = " + str(BATCH_SIZE) + "\n" +
                     "LSTM_DATA_WIDTH = " + str(LSTM_DATA_WIDTH) + "\n" +
                     "LEARNING_RATE = " + str(LEARNING_RATE) + "\n" +
                     "--------------------------------------------\n")

    aqiStatistics = pickle.load(open("dataset/aqiStatistics.pickle", "rb"))
    aqiStatistics.to_csv("result/result_" + CITY + "_" + ATTRIBUTE + ".csv", mode="a")

    with open("result/result_" + CITY + "_" + ATTRIBUTE + ".csv", "a") as result:
        result.write("--------------------------------------------\n")
        result.write("model_No,rmse,accuracy\n")
        for i in range(len(rmse_list)):
            result.write(str(i).zfill(2) + "," + str(rmse_list[i]) + "," + str(accuracy_list[i]) + "\n")

        rmse = np.average(rmse_list)
        accuracy = np.average(accuracy_list)
        result.write("average," + str(rmse) + "," + str(accuracy) + "\n")

def experiment1(LOOP, TRIAL, ATTRIBUTE, SOURCE, TARGETs, TRAIN_RATE, VALID_RATE, LSTM_DATA_WIDTH):

    '''
    Train: Source city
    Test: Source, city, Target city
    '''

    # input dimension
    dataDim = pickle.load(open("dataset/dataDim.pickle", "rb"))
    inputDim_static = dataDim["road"] + 2  # road attribute + distance + angle
    inputDim_seq_local = dataDim["meteorology"]  # meteorology attribute
    inputDim_seq_others = dataDim["meteorology"] + 1  # meteorology attribute + an aqi value

    # save input dimension
    with open("model/inputDim.pickle", "wb") as fl:
        dc = {"static": inputDim_static, "seq_local": inputDim_seq_local, "seq_others": inputDim_seq_others}
        pickle.dump(dc, fl)

    # load source and target stations
    station_source = pickle.load(open("dataset/station_"+SOURCE+".pickle", "rb"))
    station_target = []
    for i in range(len(TARGETs)):
        station_target.append(pickle.load(open("dataset/station_"+TARGETs[i]+".pickle", "rb")))

    # the number of train, validate, test datasets
    TRAIN_NUM = int(len(station_source) * TRAIN_RATE)
    VAlID_NUM = int(TRAIN_NUM * VALID_RATE)
    if VAlID_NUM < 2:
        VAlID_NUM = 2
    TEST_NUM = len(station_source)-(TRAIN_NUM+VAlID_NUM)

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    # statictics of dataset
    aqiStatistics = pickle.load(open("dataset/aqiStatistics.pickle", "rb"))
    print("# of train = "+str(TRAIN_NUM))
    print("# of valid = "+str(VAlID_NUM))
    print("# of test = "+str(TEST_NUM))
    print(aqiStatistics)

    for i in range(LOOP):

        # to evaluate
        rmse_tmp = list()
        accuracy_tmp = list()

        print("---LOOP " + str(i).zfill(2) + "---")
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
        valid = source[TRAIN_NUM:TRAIN_NUM+VAlID_NUM]
        test_source = source[TRAIN_NUM+VAlID_NUM:]
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
        print("* training ... ")
        study = optuna.create_study()
        study.optimize(objective, n_trials=TRIAL)
        print(Color.GREEN + "OK" + Color.END)

        # save best model
        model_state_dict = torch.load("tmp/" + str(study.best_trial.number).zfill(4) + "_model.pickle")
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(i).zfill(2), "model", SOURCE)
        with open(path, "wb") as pl:
            pickle.dump(model_state_dict, pl)

        # save dataset
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(i).zfill(2), "train", SOURCE)
        with open(path, "wb") as pl:
            pickle.dump(train, pl)
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(i).zfill(2), "valid", SOURCE)
        with open(path, "wb") as pl:
            pickle.dump(valid, pl)
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(i).zfill(2), "test", SOURCE)
        with open(path, "wb") as pl:
            pickle.dump(test_source, pl)
        for i in range(len(test_target)):
            path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(i).zfill(2), "test", TARGETs[i])
            with open(path, "wb") as pl:
                pickle.dump(test_target[i], pl)

        # save train log
        with open("tmp/" + str(study.best_trial.number).zfill(4) + "_log.pickle", "rb") as pl:
            log = pickle.load(pl)
            path = "log/{}_{}_{}_{}.csv".format(ATTRIBUTE, str(i).zfill(2), "log", SOURCE)
            log.to_csv(path, index=False)

        # load best model
        path = "model/{}_{}_{}_{}.pickle".format(ATTRIBUTE, str(i).zfill(2), "model", SOURCE)
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
        print("---LOOP " + str(i).zfill(2) + "---")

    # output results
    # EPOCHS = study.best_params["epochs"]
    # BATCH_SIZE = study.best_params["batch_size"]
    # LEARNING_RATE = study.best_params["learning_rate"]
    # WEIGHT_DECAY = study.best_params["weight_decay"]

    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 0.0

    path = "result/result_{}_{}.csv".format(ATTRIBUTE, SOURCE)
    with open(path, "w") as result:
        result.write("--------------------------------------------\n" +
                     "SOURCE_CITY = " + SOURCE + "\n" +
                     "TARGET_CITY = " + ",".join(TARGETs) + "\n" +
                     "MODEL_ATTRIBUTE = " + ATTRIBUTE + "\n" +
                     "TRAIN_SIZE_RATE = " + str(TRAIN_RATE) + "\n" +
                     "VARID_SIZE_RATE = " + str(VALID_RATE) + "\n" +
                     "TRAIN_NUM = " + str(TRAIN_NUM) + "\n" +
                     "VAlID_NUM = " + str(VAlID_NUM) + "\n" +
                     "TEST_NUM = " + str(TEST_NUM) + "\n" +
                     "EPOCHS_NUM = " + str(EPOCHS) + "\n" +
                     "BATCH_SIZE = " + str(BATCH_SIZE) + "\n" +
                     "LSTM_DATA_WIDTH = " + str(LSTM_DATA_WIDTH) + "\n" +
                     "LEARNING_RATE = " + str(LEARNING_RATE) + "\n" +
                     "WEIGHT_DECAY = " + str(WEIGHT_DECAY) + "\n" +
                     "--------------------------------------------\n")

    aqiStatistics.to_csv(path, mode="a")

    # to output
    rmse = np.average(np.array(rmse_list), axis=0)
    rmse = list(map(lambda x: str(x), rmse))
    rmse_list = list(map(lambda x: str(x), rmse_list))
    accuracy = np.average(np.array(accuracy_list), axis=0)
    accuracy = list(map(lambda x: str(x), accuracy))
    accuracy_list = list(map(lambda x: str(x), accuracy_list))

    with open(path, "a") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("model,{}\n".format(",".join(TARGETs)))
        for i in range(len(rmse_list)):
            result.write("{},{}\n".format(str(i).zfill(2), ",".join(rmse_list)))
        result.write("average,{}\n".format(",".join(rmse)))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("model,{}\n".format(",".join(TARGETs)))
        for i in range(len(accuracy_list)):
            result.write("{},{}\n".format(str(i).zfill(2), ",".join(accuracy_list)))
        result.write("average,{}\n".format(",".join(accuracy)))

def analysis(source, targets):
    import pandas as pd
    from pprint import pprint
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
    pprint(set(aqi_source["sid"]))
    print(aqi_source.describe())
    pd.cut(aqi_source[model_attribute], x, right=False).value_counts().sort_index().plot.bar(color='gray')
    print("////// "+source+" //////")

    for target in targets:

        print("////// "+target+" //////")
        aqi_target = pd.read_csv("database/aqi/aqi_" + target + ".csv", dtype=dtype)
        df = data_interpolate(aqi_target[[model_attribute]])
        aqi_target = pd.concat([aqi_target.drop(aqi_attribute, axis=1), df], axis=1)
        pprint(set(aqi_target["sid"]))
        print(aqi_target.describe())
        pd.cut(aqi_target[model_attribute], x, right=False).value_counts().sort_index().plot.bar(color='gray')
        print("----------------------")

        for i in range(5):

            # train
            train = pickle.load(open("model/"+source + "2" + target + "_" + model_attribute + "_"
                                     +str(i).zfill(2)+"_trainset.pickle", "rb"))
            # test: source
            test_s = pickle.load(open("model/"+source+"2"+target+"_"+model_attribute+"_"
                                     +str(i).zfill(2)+"_testset_source.pickle", "rb"))
            # test: target
            test_t = pickle.load(open("model/"+source+"2"+target+"_"+model_attribute+"_"
                                     +str(i).zfill(2)+"_testset_target.pickle", "rb"))


            aqi = aqi_source[aqi_source["sid"].isin(train)]
            print(train)
            print(aqi.describe())
            pd.cut(aqi[model_attribute], x, right=False).value_counts().sort_index().plot.bar(color='gray')

            aqi = aqi_source[aqi_source["sid"].isin(test_s)]
            print(test_s)
            print(aqi.describe())
            pd.cut(aqi[model_attribute], x, right=False).value_counts().sort_index().plot.bar(color='gray')

            aqi = aqi_target[aqi_target["sid"].isin(test_t)]
            print(test_t)
            print(aqi.describe())
            pd.cut(aqi[model_attribute], x, right=False).value_counts().sort_index().plot.bar(color='gray')

            plt.show()
            print("----------------------")
        print("////// "+target+" //////")


if __name__ == "__main__":


    ATTRIBUTE = "pm25"
    SOURCE = "beijing"
    TARGETs = ["tianjin", "guangzhou"]
    TRAIN_RATE = 0.67
    VALID_RATE = 0.1
    LSTM_DATA_WIDTH = 24
    LOOP = 5
    TRIAL = 1

    # # RE-experiment of AAAI'18
    # for CITY in [SOURCE, ] + TARGETs:
    #     makeDataset0(CITY, ATTRIBUTE, LSTM_DATA_WIDTH)
    #     experiment0(LOOP, TRIAL, ATTRIBUTE, CITY, TRAIN_RATE, VALID_RATE, LSTM_DATA_WIDTH)

    # our experiment
    makeDataset1(SOURCE, TARGETs, ATTRIBUTE, LSTM_DATA_WIDTH)
    experiment1(LOOP, TRIAL, ATTRIBUTE, SOURCE, TARGETs, TRAIN_RATE, VALID_RATE, LSTM_DATA_WIDTH)

    #reEvaluateTarget(SOURCE, TARGETs)
    #analysis(SOURCE, TARGETs)
