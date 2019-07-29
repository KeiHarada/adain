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

def experiment1(LOOP, TRIAL, ATTRIBUTE, SOURCE, TARGET, TRAIN_RATE, VALID_RATE, LSTM_DATA_WIDTH):

    '''
    Train: Source city
    Test: Source, city, Target city
    '''

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

    # load source and target stations
    station_source = pickle.load(open("dataset/stationSource.pickle", "rb"))
    station_target = pickle.load(open("dataset/stationTarget.pickle", "rb"))

    # to divide the datasets
    TRAIN_NUM = int(len(station_source) * TRAIN_RATE)
    VAlID_NUM = int(TRAIN_NUM * VALID_RATE)
    if VAlID_NUM < 2:
        VAlID_NUM = 2
    TEST_NUM = len(station_source)-(TRAIN_NUM+VAlID_NUM)

    # to evaluate
    rmse_source = list()
    accuracy_source = list()
    rmse_target = list()
    accuracy_target = list()

    # statictics of dataset
    aqiStatistics = pickle.load(open("dataset/aqiStatistics.pickle", "rb"))
    print("# of train = "+str(TRAIN_NUM))
    print("# of valid = "+str(VAlID_NUM))
    print("# of test = "+str(TEST_NUM))
    print(aqiStatistics)

    for i in range(LOOP):
        print("---LOOP " + str(i).zfill(2) + "---")
        start = time.time()

        # shuffle for LOOP
        source = station_source.copy()
        target = station_target.copy()
        random.shuffle(source)
        random.shuffle(target)

        # select train, validate, test sets
        train = source[:TRAIN_NUM]
        valid = source[TRAIN_NUM:TRAIN_NUM+VAlID_NUM]
        test_source = source[TRAIN_NUM+VAlID_NUM:]
        test_target = target[:TEST_NUM]

        # saving train, valid, test sets
        with open("tmp/trainset.pickle", "wb") as pl:
            pickle.dump(train, pl)
        with open("tmp/validset.pickle", "wb") as pl:
            pickle.dump(valid, pl)
        with open("tmp/testset_source.pickle", "wb") as pl:
            pickle.dump(test_source, pl)
        with open("tmp/testset_source.pickle", "wb") as pl:
            pickle.dump(test_target, pl)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        print("* training ... ")
        study = optuna.create_study()
        study.optimize(objective, n_trials=TRIAL)
        print(Color.GREEN + "OK" + Color.END)

        # save best model
        path = "tmp/" + str(study.best_trial.number).zfill(4) + "_model.pickle"
        model = torch.load(path)
        with open("model/" + SOURCE + "2" + TARGET + "_" + ATTRIBUTE + "_" + str(i).zfill(2) + "_model.pickle", "wb") as pl:
            pickle.dump(model, pl)

        # save dataset
        with open("model/" + SOURCE + "2" + TARGET + "_" + ATTRIBUTE + "_" + str(i).zfill(2) + "_trainset.pickle",
                  "wb") as pl:
            pickle.dump(train, pl)
        with open("model/" + SOURCE + "2" + TARGET + "_" + ATTRIBUTE + "_" + str(i).zfill(2) + "_validset.pickle",
                  "wb") as pl:
            pickle.dump(valid, pl)
        with open("model/" + SOURCE + "2" + TARGET + "_" + ATTRIBUTE + "_" + str(i).zfill(2) + "_testset_source.pickle", "wb") as pl:
            pickle.dump(test_source, pl)

        with open("model/" + SOURCE + "2" + TARGET + "_" + ATTRIBUTE + "_" + str(i).zfill(2) + "_testset_target.pickle", "wb") as pl:
            pickle.dump(test_target, pl)

        # save train log
        with open("tmp/" + str(study.best_trial.number).zfill(4) + "_log.csv", "rb") as pl:
            log = pickle.load(pl)
            log.to_csv("log/" + SOURCE + "2" + TARGET + "_" + ATTRIBUTE + "_" + str(i).zfill(2) + "_log.csv", index=False)

        # load best model
        with open("model/" + SOURCE + "2" + TARGET + "_" + ATTRIBUTE + "_" + str(i).zfill(2) + "_model.pickle", "rb") as pl:
            model_state_dict = pickle.load(pl)

        # evaluate
        print("* evaluating on source city")
        rmse, accuracy = evaluate(model_state_dict, train, test_source)
        rmse_source.append(rmse)
        accuracy_source.append(accuracy)
        print("* evaluating on target city")
        rmse, accuracy = evaluate(model_state_dict, train, test_target)
        rmse_target.append(rmse)
        accuracy_target.append(accuracy)

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

    with open("result/result_" + SOURCE + "2" + TARGET + "_" + ATTRIBUTE + ".csv", "w") as result:
        result.write("--------------------------------------------\n" +
                     "SOURCE_CITY = " + SOURCE + "\n" +
                     "TARGET_CITY = " + TARGET + "\n" +
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

    aqiStatistics.to_csv("result/result_" + SOURCE + "2" + TARGET + "_" + ATTRIBUTE + ".csv", mode="a")

    with open("result/result_" + SOURCE + "2" + TARGET + "_" + ATTRIBUTE + ".csv", "a") as result:
        result.write("--------------------------------------------\n")
        result.write("model_No,rmse_s,rmse_t,accuracy_s,accuracy_t\n")
        for i in range(len(rmse_source)):
            result.write(str(i).zfill(2) + "," + str(rmse_source[i]) + "," + str(rmse_target[i]) + ","
                         + str(accuracy_source[i]) + "," + str(accuracy_target[i]) + "\n")

        rmse_source = np.average(rmse_source)
        rmse_target = np.average(rmse_target)
        accuracy_source = np.average(accuracy_source)
        accuracy_target = np.average(accuracy_target)
        result.write("average," + str(rmse_source) + "," + str(rmse_target) + ","
                         + str(accuracy_source) + "," + str(accuracy_target) + "\n")

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

def reEvaluateTarget(source, targets):

    model_attribute = "pm25"
    # to evaluate
    rmse_source = list()
    accuracy_source = list()
    rmse_target = list()
    accuracy_target = list()

    for target in targets:

        for i in range(5):

            with open("model/" + source + "2" + target + "_" + model_attribute + "_" + str(i).zfill(2) + "_model.pickle", "rb") as pl:
                model_state_dict = pickle.load(pl)

            # train
            train = pickle.load(open("model/"+source + "2" + target + "_" + model_attribute + "_"
                                     +str(i).zfill(2)+"_trainset.pickle", "rb"))
            # test: source
            test_s = pickle.load(open("model/"+source+"2"+target+"_"+model_attribute+"_"
                                     +str(i).zfill(2)+"_testset_source.pickle", "rb"))
            # test: target
            test_t = pickle.load(open("model/"+source+"2"+target+"_"+model_attribute+"_"
                                     +str(i).zfill(2)+"_testset_target.pickle", "rb"))

            # evaluate source
            rmse, accuracy = evaluate(model_state_dict, train, test_t)
            rmse_source.append(rmse)
            accuracy_source.append(accuracy)

            # evaluate target
            rmse, accuracy = evaluate(model_state_dict, train, test_s)
            rmse_target.append(rmse)
            accuracy_target.append(accuracy)

            with open("tmp/result_" + SOURCE + "2" + TARGET + "_" + ATTRIBUTE + ".csv", "a") as result:
                result.write("--------------------------------------------\n")
                result.write("model_No,rmse_s,rmse_t,accuracy_s,accuracy_t\n")
                for i in range(len(rmse_source)):
                    result.write(str(i).zfill(2) + "," + str(rmse_source[i]) + "," + str(rmse_target[i]) + ","
                                 + str(accuracy_source[i]) + "," + str(accuracy_target[i]) + "\n")

                rmse_source = np.average(rmse_source)
                rmse_target = np.average(rmse_target)
                accuracy_source = np.average(accuracy_source)
                accuracy_target = np.average(accuracy_target)
                result.write("average," + str(rmse_source) + "," + str(rmse_target) + ","
                             + str(accuracy_source) + "," + str(accuracy_target) + "\n")


if __name__ == "__main__":


    ATTRIBUTE = "pm25"
    SOURCE = "beijing"
    TARGET = ["tianjin", "guangzhou"]
    TRAIN_RATE = 0.67
    VALID_RATE = 0.1
    LSTM_DATA_WIDTH = 24
    LOOP = 5
    TRIAL = 1

    # # RE-experiment of AAAI'18
    # for CITY in [SOURCE, ] + TARGET:
    #     makeDataset0(CITY, ATTRIBUTE, LSTM_DATA_WIDTH)
    #     experiment0(LOOP, TRIAL, ATTRIBUTE, CITY, TRAIN_RATE, VALID_RATE, LSTM_DATA_WIDTH)

    # our experiment
    # for target in TARGET:
    #     makeDataset1(SOURCE, target, ATTRIBUTE, LSTM_DATA_WIDTH)
    #     experiment1(LOOP, TRIAL, ATTRIBUTE, SOURCE, target, TRAIN_RATE, VALID_RATE, LSTM_DATA_WIDTH)

    reEvaluateTarget(SOURCE, TARGET)
    #analysis(SOURCE, TARGET)

