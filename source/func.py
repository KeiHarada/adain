# to run on server
import sys
sys.path.append("/home/harada/Documents/WorkSpace/adain")
sys.path.append("/home")

import pickle
import _pickle
import torch
import copy
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from sklearn.metrics import mean_squared_error

# from my library
from source.model import ADAIN
from source.utility import Color
from source.utility import get_dist_angle
from source.utility import calc_correct
from source.utility import get_aqi_series
from source.utility import get_meteorology_series
from source.utility import get_road_data
from source.utility import normalization
from source.utility import data_interpolate
from source.utility import weather_onehot
from source.utility import winddirection_onehot
from source.utility import EarlyStopping
from source.utility import get_activation
from source.utility import get_optimizer


def makeDataset1(source_city, target_cities, model_attribute, lstm_data_width):
    '''
    :param source_city:
    :param target_cities (list):
    :param model_attribute:
    :param lstm_data_width:
    :return:
    '''

    print("dataset ... ", end="")

    '''
    station data
    '''
    source = pd.read_csv("database/station/station_" + source_city + ".csv", dtype=object)
    with open("dataset/station_"+source_city+".pickle", "wb") as pl:
        pickle.dump(list(source["sid"]), pl)

    for target_city in target_cities:
        target = pd.read_csv("database/station/station_" + target_city + ".csv", dtype=object)
        with open("dataset/station_"+target_city+".pickle", "wb") as pl:
            pickle.dump(list(target["sid"]), pl)
        if target_city == target_cities[0]:
            station_raw = pd.concat([source, target], ignore_index=True)
        else:
            station_raw = pd.concat([station_raw, target], ignore_index=True)

    station_all = list(station_raw["sid"])

    '''
    road data
    '''
    road_attribute = ["motorway", "trunk", "others"]
    dtype = {att: "float" for att in road_attribute}
    dtype["sid"] = "object"
    source = pd.read_csv("database/road/road_" + source_city + ".csv", dtype=dtype)
    for target_city in target_cities:
        target = pd.read_csv("database/road/road_" + target_city + ".csv", dtype=dtype)
        if target_city == target_cities[0]:
            road_raw = pd.concat([source, target], ignore_index=True)
        else:
            road_raw = pd.concat([road_raw, target], ignore_index=True)
    df = normalization(road_raw[road_attribute])
    road_raw = pd.concat([road_raw.drop(road_attribute, axis=1), df], axis=1)

    '''
    meteorology data
    '''
    meteorology_attribute = ["weather", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
    dtype = {att: "float" for att in meteorology_attribute}
    dtype["did"], dtype["time"] = "object", "object"
    source = pd.read_csv("database/meteorology/meteorology_" + source_city + ".csv", dtype=dtype)
    for target_city in target_cities:
        target = pd.read_csv("database/meteorology/meteorology_" + target_city + ".csv", dtype=dtype)
        if target_city == target_cities[0]:
            meteorology_raw = pd.concat([source, target], ignore_index=True)
        else:
            meteorology_raw = pd.concat([meteorology_raw, target], ignore_index=True)
    meteorology_attribute = ["temperature", "pressure", "humidity", "wind_speed"]
    df = normalization(data_interpolate(meteorology_raw[meteorology_attribute]))
    meteorology_raw = pd.concat([meteorology_raw.drop(meteorology_attribute, axis=1), df], axis=1)
    df, columns = weather_onehot(meteorology_raw["weather"])
    meteorology_raw = pd.concat([meteorology_raw.drop(["weather"], axis=1), df], axis=1)
    meteorology_attribute += columns
    df, columns = winddirection_onehot(meteorology_raw["wind_direction"])
    meteorology_raw = pd.concat([meteorology_raw.drop(["wind_direction"], axis=1), df], axis=1)
    meteorology_attribute += columns

    '''
    aqi data
    '''
    aqi_attribute = ["pm25", "pm10", "no2", "co", "o3", "so2"]
    dtype = {att: "float" for att in aqi_attribute}
    dtype["sid"], dtype["time"] = "object", "object"
    source = pd.read_csv("database/aqi/aqi_" + source_city + ".csv", dtype=dtype)
    for target_city in target_cities:
        target = pd.read_csv("database/aqi/aqi_" + target_city + ".csv", dtype=dtype)
        if target_city == target_cities[0]:
            aqi_raw = pd.concat([source, target], ignore_index=True)
        else:
            aqi_raw = pd.concat([aqi_raw, target], ignore_index=True)
    df = data_interpolate(aqi_raw[[model_attribute]])
    aqi_raw = pd.concat([aqi_raw.drop(aqi_attribute, axis=1), df], axis=1)
    with open("dataset/aqiStatistics.pickle", "wb") as pl:
        pickle.dump(aqi_raw.describe(), pl)

    '''
    make dataset
    '''
    staticData, seqData_m, seqData_a, targetData = dict(), dict(), dict(), dict()
    for sid in station_all:

        '''
        static data
            * road network data
            * poi (TODO)

        format
            r = [motorway, trunk, others]
            X = [ r ]
        '''
        recode = [float(get_road_data(data=road_raw, sid=sid, attribute=att)) for att in road_attribute]
        staticData[sid] = [recode]

        '''
        sequence data
            * meteorological data
            * aqi data

        format
            r_t = ["weather", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
            R_p = [r_t, ..., r_t+p]
            X = [ R_p, R_p+1, ..., R_n ]

        format
            r_t = ["pm25" or  "pm10" or "no2" or "co" or "o3" or "so2"]
            R_p = [r_t, ..., r_t+p]
            X = [ R_p, R_p+1, ..., R_n ]
        '''
        did = list(station_raw[station_raw["sid"] == sid]["did"])[0]
        meteorology_data = {att: get_meteorology_series(data=meteorology_raw, did=did, attribute=att)
                            for att in meteorology_attribute}
        aqi_data = get_aqi_series(data=aqi_raw, sid=sid, attribute=model_attribute)

        seqData_m[sid] = []
        seqData_a[sid] = []
        terminal = len(meteorology_data[meteorology_attribute[0]])
        start = 0
        end = lstm_data_width
        while end <= terminal:
            recode_m_p = []
            recode_a_p = []
            for t in range(start, end):
                recode_m_t = []
                for att in meteorology_attribute:
                    recode_m_t.append(meteorology_data[att][t])
                recode_m_p.append(recode_m_t)
                recode_a_p.append([aqi_data[t]])
            seqData_m[sid].append(recode_m_p)
            seqData_a[sid].append(recode_a_p)
            start += 1
            end += 1

        '''
        target data
            * aqi data

        format
            aqi = "pm25" or  "pm10" or "no2" or "co" or "o3" or "so2"
            X = [ [aqi_p], [aqi_p+1], ..., [aqi_n] ]
        '''
        targetData[sid] = []
        for t in range(lstm_data_width - 1, len(aqi_data)):
            targetData[sid].append([aqi_data[t]])

    # saving
    with open("dataset/dataDim.pickle", "wb") as pl:
        dc = {"road": len(road_attribute),
              "meteorology": len(meteorology_attribute)}
        pickle.dump(dc, pl)

    with open("dataset/stationData.pickle", "wb") as pl:
        pickle.dump(station_raw, pl)

    with open("dataset/staticData.pickle", "wb") as pl:
        pickle.dump(staticData, pl)

    with open("dataset/meteorologyData.pickle", "wb") as pl:
        pickle.dump(seqData_m, pl)

    with open("dataset/aqiData.pickle", "wb") as pl:
        pickle.dump(seqData_a, pl)

    with open("dataset/targetData.pickle", "wb") as pl:
        pickle.dump(targetData, pl)

    print(Color.GREEN + "OK" + Color.END)

def makeDataset0(city_name, model_attribute, lstm_data_width):
    '''
    :param city_name:
    :param model_attribute:
    :param lstm_data_width:
    :return:
    '''

    print("dataset ... ", end="")
    '''
    station data
    '''
    station_raw = pd.read_csv("database/station/station_" + city_name + ".csv", dtype=object)
    station_all = list(station_raw["sid"])

    '''
    road data
    '''
    road_attribute = ["motorway", "trunk", "others"]
    dtype = {att: "float" for att in road_attribute}
    dtype["sid"] = "object"
    road_raw = pd.read_csv("database/road/road_" + city_name + ".csv", dtype=dtype)
    df = normalization(road_raw[road_attribute])
    road_raw = pd.concat([road_raw.drop(road_attribute, axis=1), df], axis=1)

    '''
    meteorology data
    '''
    meteorology_attribute = ["weather", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
    dtype = {att: "float" for att in meteorology_attribute}
    dtype["did"], dtype["time"] = "object", "object"
    meteorology_raw = pd.read_csv("database/meteorology/meteorology_" + city_name + ".csv", dtype=dtype)
    meteorology_attribute = ["temperature", "pressure", "humidity", "wind_speed"]
    df = normalization(data_interpolate(meteorology_raw[meteorology_attribute]))
    meteorology_raw = pd.concat([meteorology_raw.drop(meteorology_attribute, axis=1), df], axis=1)
    df, columns = weather_onehot(meteorology_raw["weather"])
    meteorology_raw = pd.concat([meteorology_raw.drop(["weather"], axis=1), df], axis=1)
    meteorology_attribute += columns
    df, columns = winddirection_onehot(meteorology_raw["wind_direction"])
    meteorology_raw = pd.concat([meteorology_raw.drop(["wind_direction"], axis=1), df], axis=1)
    meteorology_attribute += columns

    '''
    aqi data
    '''
    aqi_attribute = ["pm25", "pm10", "no2", "co", "o3", "so2"]
    dtype = {att: "float" for att in aqi_attribute}
    dtype["sid"], dtype["time"] = "object", "object"
    aqi_raw = pd.read_csv("database/aqi/aqi_" + city_name + ".csv", dtype=dtype)
    df = data_interpolate(aqi_raw[[model_attribute]])
    aqi_raw = pd.concat([aqi_raw.drop(aqi_attribute, axis=1), df], axis=1)
    with open("dataset/aqiStatistics.pickle", "wb") as pl:
        pickle.dump(aqi_raw.describe(), pl)

    '''
    make dataset
    '''
    staticData, seqData_m, seqData_a, targetData = dict(), dict(), dict(), dict()
    for sid in station_all:

        '''
        static data
            * road network data
            * poi (TODO)

        format
            r = [motorway, trunk, others]
            X = [ r ]
        '''
        recode = [float(get_road_data(data=road_raw, sid=sid, attribute=att)) for att in road_attribute]
        staticData[sid] = [recode]

        '''
        sequence data
            * meteorological data
            * aqi data

        format
            r_t = ["weather", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
            R_p = [r_t, ..., r_t+p]
            X = [ R_p, R_p+1, ..., R_n ]

        format
            r_t = ["pm25" or  "pm10" or "no2" or "co" or "o3" or "so2"]
            R_p = [r_t, ..., r_t+p]
            X = [ R_p, R_p+1, ..., R_n ]
        '''
        did = list(station_raw[station_raw["sid"] == sid]["did"])[0]
        meteorology_data = {att: get_meteorology_series(data=meteorology_raw, did=did, attribute=att)
                            for att in meteorology_attribute}
        aqi_data = get_aqi_series(data=aqi_raw, sid=sid, attribute=model_attribute)

        seqData_m[sid] = []
        seqData_a[sid] = []
        terminal = len(meteorology_data[meteorology_attribute[0]])
        start = 0
        end = lstm_data_width
        while end <= terminal:
            recode_m_p = []
            recode_a_p = []
            for t in range(start, end):
                recode_m_t = []
                for att in meteorology_attribute:
                    recode_m_t.append(meteorology_data[att][t])
                recode_m_p.append(recode_m_t)
                recode_a_p.append([aqi_data[t]])
            seqData_m[sid].append(recode_m_p)
            seqData_a[sid].append(recode_a_p)
            start += 1
            end += 1

        '''
        target data
            * aqi data

        format
            aqi = "pm25" or  "pm10" or "no2" or "co" or "o3" or "so2"
            X = [ [aqi_p], [aqi_p+1], ..., [aqi_n] ]
        '''
        targetData[sid] = []
        for t in range(lstm_data_width - 1, len(aqi_data)):
            targetData[sid].append([aqi_data[t]])

    # saving
    with open("dataset/dataDim.pickle", "wb") as pl:
        dc = {"road": len(road_attribute),
              "meteorology": len(meteorology_attribute)}
        pickle.dump(dc, pl)

    with open("dataset/stationAll.pickle", "wb") as pl:
        pickle.dump(station_all, pl)

    with open("dataset/stationData.pickle", "wb") as pl:
        pickle.dump(station_raw, pl)

    with open("dataset/staticData.pickle", "wb") as pl:
        pickle.dump(staticData, pl)

    with open("dataset/meteorologyData.pickle", "wb") as pl:
        pickle.dump(seqData_m, pl)

    with open("dataset/aqiData.pickle", "wb") as pl:
        pickle.dump(seqData_a, pl)

    with open("dataset/targetData.pickle", "wb") as pl:
        pickle.dump(targetData, pl)

    print(Color.GREEN + "OK" + Color.END)

def makeTestBatch(item, batch_length):

    '''
    :param item:
    :param batch_length:
    :return: a list of batch
    '''

    # input
    local_static, local_seq, others_static, others_seq, target = item

    # output
    batch = list()

    # running condition
    if len(target) < batch_length:
        exit("batch length is too large. it must be not more than %d" % (len(target)))

    '''
    a batch = (batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target)
    '''
    offset = 0
    for i in range(batch_length):

        batch_local_static = []
        batch_local_seq = []
        batch_others_static = [[] for _ in range(len(others_static))] # [[]]*len(n) >> [[], ..., []] だと同じ場所を参照する配列がn個できるので注意
        batch_others_seq = [[] for _ in range(len(others_seq))]
        batch_target = []

        batch_size = len(target) // batch_length

        for j in range(batch_size):
            batch_local_static.append(local_static[0])
            batch_local_seq.append(local_seq[offset + j])
            batch_target.append(target[offset + j])
            for k in range(len(others_static)):
                batch_others_static[k].append(others_static[k][0])
                batch_others_seq[k].append(others_seq[k][offset + j])

        batch.append((batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target))

        offset += batch_size

    return batch

def makeRandomBatch(item, batch_length, batch_size):

    '''
    :param item: a set of (local_static, local_seq, others_static, others_seq, target)
    :param batch_length:
    :param batch_size:
    :return: a list of batches
    '''

    # input
    local_static, local_seq, others_static, others_seq, target = item

    # output
    batch = list()
    '''
    a batch = (batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target)
    '''
    for i in range(batch_length):

        batch_local_static = []
        batch_local_seq = []
        batch_others_static = [[] for _ in range(len(others_static))] # [[]]*len(n) >> [[], ..., []] だと同じ場所を参照する配列がn個できるので注意
        batch_others_seq = [[] for _ in range(len(others_seq))]
        batch_target = []

        for j in range(batch_size):
            idx = np.random.randint(0, len(target) - 1)
            batch_local_static.append(local_static[0])
            batch_local_seq.append(local_seq[idx])
            batch_target.append(target[idx])
            for k in range(len(batch_others_static)):
                batch_others_static[k].append(others_static[k][0])
                batch_others_seq[k].append(others_seq[k][idx])

        batch.append((batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target))

    return batch

def loadTrainData(station_train):

    '''
    :param station_train): a list of station ids
    :return: a list of trainData
    '''

    # raw data
    stationData = pickle.load(open("dataset/stationData.pickle", "rb"))
    staticData = pickle.load(open("dataset/staticData.pickle", "rb"))
    meteorologyData = pickle.load(open("dataset/meteorologyData.pickle", "rb"))
    aqiData = pickle.load(open("dataset/aqiData.pickle", "rb"))
    targetData = pickle.load(open("dataset/targetData.pickle", "rb"))

    # output
    trainData = list()

    # make a list of train data
    '''
    a train data = (local_static, local_seq, others_static, others_seq, target)
    '''
    for station_local in station_train:

        station_others = station_train.copy()
        station_others.remove(station_local)

        # station local
        local_static = [staticData[station_local][0] + [0, 0]]
        local_seq = meteorologyData[station_local]

        # lat, lon of local station
        lat_local = float(stationData[stationData["sid"] == station_local]["lat"])
        lon_local = float(stationData[stationData["sid"] == station_local]["lon"])

        # distance and angle data
        distance = list()
        angle = list()
        for sid in station_others:
            lat = float(stationData[stationData["sid"] == sid]["lat"])
            lon = float(stationData[stationData["sid"] == sid]["lon"])
            result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
            distance.append(result["distance"])
            angle.append(result["azimuth1"])

        # normalization
        maximum = max(distance)
        minimum = min(distance)
        distance = list(map(lambda x: (x - minimum) / (maximum - minimum), distance))
        maximum = max(angle)
        minimum = min(angle)
        angle = list(map(lambda x: (x - minimum) / (maximum - minimum), angle))

        # make dictionary "geo"
        geo = dict()
        idx = 0
        for sid in station_others:
            geo[sid] = dict()
            geo[sid]["distance"] = distance[idx]
            geo[sid]["angle"] = angle[idx]
            idx += 1

        # station others
        others_static = []
        others_seq = []
        for sid in station_others:

            stat = _pickle.loads(_pickle.dumps(staticData[sid], -1)) # _pickleを使った高速コピー
            stat[0].append(geo[sid]["distance"])
            stat[0].append(geo[sid]["angle"])
            others_static.append(stat)

            m = _pickle.loads(_pickle.dumps(meteorologyData[sid], -1)) # _pickleを使った高速コピー
            a = _pickle.loads(_pickle.dumps(aqiData[sid], -1)) # _pickleを使った高速コピー
            for i in range(len(m)):
                for j in range(len(m[i])):
                    m[i][j] += a[i][j]
            others_seq.append(m)

        # target
        target = targetData[station_local]

        trainData.append((local_static, local_seq, others_static, others_seq, target))


    return trainData

def loadTestData(station_test, station_train):

    '''
    :param station_test:
    :param station_train:
    :return:
    '''

    # raw data
    stationData = pickle.load(open("dataset/stationData.pickle", "rb"))
    staticData = pickle.load(open("dataset/staticData.pickle", "rb"))
    meteorologyData = pickle.load(open("dataset/meteorologyData.pickle", "rb"))
    aqiData = pickle.load(open("dataset/aqiData.pickle", "rb"))
    targetData = pickle.load(open("dataset/targetData.pickle", "rb"))

    # output
    testData = list()

    # make a list of train data
    '''
    a test data = (local_static, local_seq, others_static, others_seq, target)
    '''
    for station_local in station_test:

        for station_removed in station_train:

            station_others = station_train.copy()
            station_others.remove(station_removed)

            # station local
            local_static = [staticData[station_local][0] + [0, 0]]
            local_seq = meteorologyData[station_local]

            # lat, lon of local station
            lat_local = float(stationData[stationData["sid"] == station_local]["lat"])
            lon_local = float(stationData[stationData["sid"] == station_local]["lon"])

            # distance and angle data
            distance = list()
            angle = list()
            for sid in station_others:
                lat = float(stationData[stationData["sid"] == sid]["lat"])
                lon = float(stationData[stationData["sid"] == sid]["lon"])
                result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
                distance.append(result["distance"])
                angle.append(result["azimuth1"])

            # normalization
            maximum = max(distance)
            minimum = min(distance)
            distance = list(map(lambda x: (x - minimum) / (maximum - minimum), distance))
            maximum = max(angle)
            minimum = min(angle)
            angle = list(map(lambda x: (x - minimum) / (maximum - minimum), angle))

            # make dictionary "geo"
            geo = dict()
            idx = 0
            for sid in station_others:
                geo[sid] = dict()
                geo[sid]["distance"] = distance[idx]
                geo[sid]["angle"] = angle[idx]
                idx += 1

            # station others
            others_static = []
            others_seq = []
            for sid in station_others:

                stat = _pickle.loads(_pickle.dumps(staticData[sid], -1))  # _pickleを使った高速コピー
                stat[0].append(geo[sid]["distance"])
                stat[0].append(geo[sid]["angle"])
                others_static.append(stat)

                m = _pickle.loads(_pickle.dumps(meteorologyData[sid], -1))  # _pickleを使った高速コピー
                a = _pickle.loads(_pickle.dumps(aqiData[sid], -1))  # _pickleを使った高速コピー
                for i in range(len(m)):
                    for j in range(len(m[i])):
                        m[i][j] += a[i][j]
                others_seq.append(m)

            # target
            target = targetData[station_local]

            testData.append((local_static, local_seq, others_static, others_seq, target))

    return testData

def validate(model, validData):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for evaluation
    result = []
    result_label = []

    # the number to divide the whole of the test data into min-batches
    batch_length = 10

    for item in validData:

        for itr in makeTestBatch(item, batch_length):

            batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target = itr

            # to tensor
            batch_local_static = torch.tensor(batch_local_static).to(device)
            batch_local_seq = torch.tensor(batch_local_seq).to(device)
            batch_others_static = list(map(lambda x: torch.tensor(x).to(device), batch_others_static))
            batch_others_seq = list(map(lambda x: torch.tensor(x).to(device), batch_others_seq))

            # predict
            pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq)
            pred = pred.to("cpu")

            # evaluate
            pred = list(map(lambda x: x[0], pred.data.numpy()))
            batch_target = list(map(lambda x: x[0], batch_target))
            result += pred
            result_label += batch_target

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)

    return rmse, accuracy

def objective(trial):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyper parameters for tuning
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    # epochs = trial.suggest_discrete_uniform("epochs", 1, 5, 1)
    # lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    # wd = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    # hyper parameters for constance
    batch_size = 256
    epochs = 50
    lr = 0.001
    wd = 0.0

    # input dimension
    inputDim = pickle.load(open("model/inputDim.pickle", "rb"))

    # model
    model = ADAIN(inputDim_static=inputDim["static"],
                  inputDim_seq_local=inputDim["seq_local"],
                  inputDim_seq_others=inputDim["seq_others"])

    # GPU or CPU
    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # evaluation function
    criterion = nn.MSELoss()

    # train data
    station_train = pickle.load(open("tmp/trainset.pickle", "rb"))
    station_valid = pickle.load(open("tmp/validset.pickle", "rb"))

    # initialize the early stopping object
    patience = 10
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # log
    logs = []

    # load data
    print("data loading ...", end="")
    trainData = loadTrainData(station_train)
    validData = loadTestData(station_valid, station_train)
    print(Color.GREEN + "OK" + Color.END)

    for step in range(int(epochs)):

        step_loss = []

        running = 0
        maximum = int(len(trainData))

        # train
        for item in trainData:

            running +=1
            print("\t|- running loss %d / %d : " % (running, maximum), end="")

            running_loss = []
            batch_length = len(item[4]) # len(item[4]): len(target) --> batch_size

            for itr in makeRandomBatch(item, batch_length, batch_size):

                batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target = itr

                # to tensor
                batch_local_static = torch.tensor(batch_local_static).to(device)
                batch_local_seq = torch.tensor(batch_local_seq).to(device)
                batch_others_static = list(map(lambda x: torch.tensor(x).to(device), batch_others_static))
                batch_others_seq = list(map(lambda x: torch.tensor(x).to(device), batch_others_seq))
                batch_target = torch.tensor(batch_target).to(device)

                # predict
                pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq)

                loss = criterion(pred, batch_target)
                loss.backward()
                optimizer.step()
                running_loss.append(np.sqrt(loss.item()))

            running_loss = np.average(running_loss)
            step_loss.append(running_loss)
            print("%.10f" % (running_loss))

        step_loss = np.average(step_loss)
        print("\t\t|- epoch %d loss: %.10f" % (step + 1, step_loss))

        # validate
        print("\t\t|- validation : ", end="")
        model.eval()
        rmse, accuracy = validate(model, validData)
        model.train()
        log = {'epoch': step, 'validation rmse': rmse, 'validation accuracy': accuracy}
        logs.append(log)
        print("rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

        # early stopping
        early_stopping(rmse, model)
        if early_stopping.early_stop:
            print("\t\tEarly stopping")
            break

    # load the last checkpoint after early stopping
    model.load_state_dict(torch.load("tmp/checkpoint.pt"))

    # save model
    trial_num = trial.number
    with open("tmp/" + str(trial_num).zfill(4) + "_model.pickle", "wb") as pl:
        torch.save(model.state_dict(), pl)

    # save logs
    logs = pd.DataFrame(logs)
    with open("tmp/" + str(trial_num).zfill(4) + "_log.pickle", "wb") as pl:
        pickle.dump(logs, pl)

    return rmse


def evaluate(model_state_dict, station_train, station_test):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # input dimension
    inputDim = pickle.load(open("model/inputDim.pickle", "rb"))

    # model
    model = ADAIN(inputDim_static=inputDim["static"],
                  inputDim_seq_local=inputDim["seq_local"],
                  inputDim_seq_others=inputDim["seq_others"])

    model.load_state_dict(model_state_dict)
    model = model.to(device)

    # evaluate mode
    model.eval()

    # for evaluation
    result = []
    result_label = []

    # the number to divide the whole of the test data into min-batches
    batch_length = 5

    for s in station_test:

        # load data
        print("data loading ....")
        testData = loadTestData(list(s), station_train)
        print(Color.GREEN + "OK" + Color.END)

        for item in testData:

            for itr in makeTestBatch(item, batch_length):
                batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target = itr

                # to tensor
                batch_local_static = torch.tensor(batch_local_static).to(device)
                batch_local_seq = torch.tensor(batch_local_seq).to(device)
                batch_others_static = list(map(lambda x: torch.tensor(x).to(device), batch_others_static))
                batch_others_seq = list(map(lambda x: torch.tensor(x).to(device), batch_others_seq))

                # predict
                pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq)
                pred = pred.to("cpu")

                # evaluate
                pred = list(map(lambda x: x[0], pred.data.numpy()))
                batch_target = list(map(lambda x: x[0], batch_target))
                result += pred
                result_label += batch_target

            print("\t|- iteration %d / %d" % (int(testData.index(item))+1, int(len(testData))))

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)
    print("rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

    return rmse, accuracy