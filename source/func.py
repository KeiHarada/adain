import pickle
import torch
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


def makeDataset_st(source_city, target_city, model_attribute, lstm_data_width):
    '''
    :param source_city:
    :param model_attribute:
    :param lstm_data_width:
    :return:
    '''

    print("dataset ... ", end="")
    '''
    station data
    '''
    source = pd.read_csv("database/station/station_" + source_city + ".csv", dtype=object)
    target = pd.read_csv("database/station/station_" + target_city + ".csv", dtype=object)

    with open("dataset/stationSource.pickle", "wb") as pl:
        pickle.dump(list(source["sid"]), pl)
    with open("dataset/stationTarget.pickle", "wb") as pl:
        pickle.dump(list(target["sid"]), pl)

    station_raw = pd.concat([source, target], ignore_index=True)
    station_all = list(station_raw["sid"])

    '''
    road data
    '''
    road_attribute = ["motorway", "trunk", "others"]
    dtype = {att: "float" for att in road_attribute}
    dtype["sid"] = "object"
    source = pd.read_csv("database/road/road_" + source_city + ".csv", dtype=dtype)
    target = pd.read_csv("database/road/road_" + target_city + ".csv", dtype=dtype)
    road_raw = pd.concat([source, target], ignore_index=True)

    '''
    meteorology data
    '''
    meteorology_attribute = ["weather", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
    dtype = {att: "float" for att in meteorology_attribute}
    dtype["did"], dtype["time"] = "object", "object"
    source = pd.read_csv("database/meteorology/meteorology_" + source_city + ".csv", dtype=dtype)
    target = pd.read_csv("database/meteorology/meteorology_" + target_city + ".csv", dtype=dtype)
    meteorology_raw = pd.concat([source, target], ignore_index=True)
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
    target = pd.read_csv("database/aqi/aqi_" + target_city + ".csv", dtype=dtype)
    aqi_raw = pd.concat([source, target], ignore_index=True)
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

def makeDataset(city_name, model_attribute, lstm_data_width):
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

def makeTestBatch(local_static, local_seq, others_static, others_seq, target, divide_num):
    batch_local_static = []
    batch_local_seq = []
    batch_others_static = []
    batch_others_seq = []
    batch_target = []

    batch_size = len(local_seq) // divide_num
    for i in range(divide_num):
        if i == divide_num-1:
            batch_local_static.append(torch.tensor([local_static[0]] * len(local_seq[i * batch_size:])))
            batch_local_seq.append(torch.tensor(local_seq[i * batch_size:]))
            batch_target.append(torch.tensor(target[i*batch_size:]))
            batch_others_static_i = [[] for _ in range(len(others_static))]
            batch_others_seq_i = [[] for _ in range(len(others_seq))]
            for j in range(len(others_static)):
                batch_others_static_i[j] = torch.tensor([others_static[j][0]] * len(local_seq[i * batch_size:]))
                batch_others_seq_i[j] = torch.tensor(others_seq[j][i * batch_size:])
            batch_others_static.append(batch_others_static_i)
            batch_others_seq.append(batch_others_seq_i)
        else:
            batch_local_static.append(torch.tensor([local_static[0]] * batch_size))
            batch_local_seq.append(torch.tensor(local_seq[i * batch_size: (i + 1) * batch_size]))
            batch_target.append(torch.tensor(target[i*batch_size: (i+1)*batch_size]))
            batch_others_static_i = [[] for _ in range(len(others_static))]
            batch_others_seq_i = [[] for _ in range(len(others_seq))]
            for j in range(len(others_static)):
                batch_others_static_i[j] = torch.tensor([others_static[j][0]] * batch_size)
                batch_others_seq_i[j] = torch.tensor(others_seq[j][i * batch_size: (i + 1) * batch_size])
            batch_others_static.append(batch_others_static_i)
            batch_others_seq.append(batch_others_seq_i)


    return batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target

def makeRandomBatch(local_static, local_seq, others_static, others_seq, target, batch_size):
    batch_local_static = []
    batch_local_seq = []
    batch_others_static = [[] for _ in range(len(others_static))] # [[]]*len(n) >> [[], ..., []] だと同じ場所を参照する配列がn個できるので注意
    batch_others_seq = [[] for _ in range(len(others_seq))]
    batch_target = []

    for _ in range(batch_size):
        idx = np.random.randint(0, len(target) - 1)
        batch_local_static.append(local_static[0])
        batch_local_seq.append(local_seq[idx])
        batch_target.append(target[idx])
        for i in range(len(batch_others_static)):
            batch_others_static[i].append(others_static[i][0])
            batch_others_seq[i].append(others_seq[i][idx])

    batch_local_static = torch.tensor(batch_local_static)
    batch_local_seq = torch.tensor(batch_local_seq)
    batch_others_static = list(map(lambda x: torch.tensor(x), batch_others_static))
    batch_others_seq = list(map(lambda x: torch.tensor(x), batch_others_seq))
    batch_target = torch.tensor(batch_target)

    return batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target

def loadData(sid_local, sid_others):

    # raw data
    stationData = pickle.load(open("dataset/stationData.pickle", "rb"))
    staticData = pickle.load(open("dataset/staticData.pickle", "rb"))
    meteorologyData = pickle.load(open("dataset/meteorologyData.pickle", "rb"))
    aqiData = pickle.load(open("dataset/aqiData.pickle", "rb"))
    targetData = pickle.load(open("dataset/targetData.pickle", "rb"))

    # local
    train_local_static = [staticData[sid_local][0] + [0, 0]]
    train_local_seq = meteorologyData[sid_local]

    # lat, lon of local station
    lat_local = float(stationData[stationData["sid"] == sid_local]["lat"])
    lon_local = float(stationData[stationData["sid"] == sid_local]["lon"])

    # distance and angle data
    distance = list()
    angle = list()
    for sid in sid_others:
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

    # make dictionary
    geo = dict()
    idx = 0
    for sid in sid_others:
        geo[sid] = dict()
        geo[sid]["distance"] = distance[idx]
        geo[sid]["angle"] = angle[idx]
        idx += 1

    # other stations
    train_others_static = []
    train_others_seq = []
    for sid in sid_others:

        stat = staticData[sid]
        stat[0].append(geo[sid]["distance"])
        stat[0].append(geo[sid]["angle"])
        train_others_static.append(stat)

        m = meteorologyData[sid]
        a = aqiData[sid]
        for i in range(len(m)):
            for j in range(len(m[i])):
                m[i][j] += a[i][j]
        train_others_seq.append(m)

    # target
    target = targetData[sid_local]

    return train_local_static, train_local_seq, train_others_static, train_others_seq, target

def validate(model, criterion, station_valid, station_train):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # the number to divide the whole of the test data into min-batches
    divide_num = 50

    # for evaluation
    valid_loss = []

    for station_local in station_valid:

        for station_remove in station_train:
            station_others = station_train.copy()
            station_others.remove(station_remove)

            # get train and target data
            test_local_static, \
            test_local_seq, \
            test_others_static, \
            test_others_seq, \
            target = loadData(station_local, station_others)

            # divide the test data into some sub-data
            batch_local_static, \
            batch_local_seq, \
            batch_others_static, \
            batch_others_seq, \
            batch_target = makeTestBatch(test_local_static,
                                         test_local_seq,
                                         test_others_static,
                                         test_others_seq,
                                         target, divide_num)

            for i in range(divide_num):
                x_local_static = batch_local_static[i]
                x_local_seq = batch_local_seq[i]
                x_others_static = batch_others_static[i]
                x_others_seq = batch_others_seq[i]
                y_label = batch_target[i]

                # GPU or CPU
                x_local_static = x_local_static.to(device)
                x_local_seq = x_local_seq.to(device)
                x_others_static = list(map(lambda x: x.to(device), x_others_static))
                x_others_seq = list(map(lambda x: x.to(device), x_others_seq))
                y_label = y_label.to(device)

                y = model(x_local_static, x_local_seq, x_others_static, x_others_seq)
                valid_loss.append(criterion(y, y_label).item())

    # RMSE
    return np.sqrt(np.average(valid_loss))


def objective(trial):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # to tune
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
    # epochs = trial.suggest_discrete_uniform("epochs", 1, 5, 1)
    # lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    # wd = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    # no tune
    batch_size = 32
    epochs = 50
    lr = 0.01
    wd = 0.0

    # input dimension
    inputDim = pickle.load(open("model/inputDim.pickle", "rb"))

    # model
    model = ADAIN(inputDim_static=inputDim["static"],
                  inputDim_seq_local=inputDim["seq_local"],
                  inputDim_seq_others=inputDim["seq_others"])

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # evaluation function
    criterion = nn.MSELoss()

    # training
    station_train = pickle.load(open("tmp/trainset.pickle", "rb"))
    station_valid = pickle.load(open("tmp/validset.pickle", "rb"))

    # initialize the early stopping object
    patience = int(int(epochs)*0.2)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for step in range(int(epochs)):

        step_loss = []

        for station_local in station_train:

            running_loss = []

            # divide stations into local and others
            station_others = station_train.copy()
            station_others.remove(station_local)

            # get train and target data
            train_local_static, \
            train_local_seq, \
            train_others_static, \
            train_others_seq, \
            target = loadData(station_local, station_others)

            # training
            train_data_size = len(train_local_seq)
            iterations = train_data_size // batch_size
            for _ in range(iterations):
                optimizer.zero_grad()

                # make min-batch
                x_local_static, \
                x_local_seq, \
                x_others_static, \
                x_others_seq, \
                y_label = makeRandomBatch(train_local_static,
                                          train_local_seq,
                                          train_others_static,
                                          train_others_seq,
                                          target,
                                          batch_size)

                # GPU or CPU
                x_local_static = x_local_static.to(device)
                x_local_seq = x_local_seq.to(device)
                x_others_static = list(map(lambda x: x.to(device), x_others_static))
                x_others_seq = list(map(lambda x: x.to(device), x_others_seq))
                y_label = y_label.to(device)

                y = model(x_local_static, x_local_seq, x_others_static, x_others_seq)

                loss = criterion(y, y_label)
                loss.backward()
                optimizer.step()
                running_loss.append(np.sqrt(loss.item()))

            running_loss = np.average(running_loss)
            step_loss.append(running_loss)
            print("\t|- local %d loss: %.10f" % (station_train.index(station_local) + 1, running_loss))

        step_loss = np.average(step_loss)
        print("\t\t|- epoch %d loss: %.10f" % (step + 1, step_loss))

        # validation
        model.eval()
        valid_loss = validate(model, criterion, station_valid, station_train)
        model.train()
        print("\t\t|- validation loss: %.10f" % (valid_loss))

        # early stopping
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("\t\tEarly stopping")
            break

    # load the last checkpoint after early stopping
    model.load_state_dict(torch.load("tmp/checkpoint.pt"))

    # saving model
    trial_num = trial.number
    with open("tmp/" + str(trial_num).zfill(4) + "_model.pickle", "wb") as pl:
        torch.save(model.state_dict(), pl)

    return valid_loss


def evaluate(best_trial, station_train, station_test):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # input dimension
    inputDim = pickle.load(open("model/inputDim.pickle", "rb"))

    # model
    model = ADAIN(inputDim_static=inputDim["static"],
                  inputDim_seq_local=inputDim["seq_local"],
                  inputDim_seq_others=inputDim["seq_others"])

    model_path = "tmp/" + str(best_trial.number).zfill(4) + "_model.pickle"
    model.load_state_dict(torch.load(model_path))

    # evaluate mode
    model.eval()

    # the number to divide the whole of the test data into min-batches
    divide_num = 50

    # for evaluation
    result = []
    result_label = []

    itr = 0
    ITR = len(station_test) * len(station_train) * divide_num
    for station_local in station_test:

        for station_remove in station_train:
            station_others = station_train.copy()
            station_others.remove(station_remove)

            # get train and target data
            test_local_static, \
            test_local_seq, \
            test_others_static, \
            test_others_seq, \
            target = loadData(station_local, station_others)

            # divide the test data into some sub-data
            batch_local_static, \
            batch_local_seq, \
            batch_others_static, \
            batch_others_seq, \
            batch_target = makeTestBatch(test_local_static,
                                         test_local_seq,
                                         test_others_static,
                                         test_others_seq,
                                         target,
                                         divide_num)

            for i in range(divide_num):
                x_local_static = batch_local_static[i]
                x_local_seq = batch_local_seq[i]
                x_others_static = batch_others_static[i]
                x_others_seq = batch_others_seq[i]
                y_label = batch_target[i]

                # GPU or CPU
                x_local_static = x_local_static.to(device)
                x_local_seq = x_local_seq.to(device)
                x_others_static = list(map(lambda x: x.to(device), x_others_static))
                x_others_seq = list(map(lambda x: x.to(device), x_others_seq))

                y = model(x_local_static, x_local_seq, x_others_static, x_others_seq)
                y = y.to("cpu")

                # evaluate
                y = list(map(lambda x: x[0], y.data.numpy()))
                y_label = list(map(lambda x: x[0], y_label.data.numpy()))
                result += y
                result_label += y_label
                itr += 1

        print("\t|- iteration %d / %d" % (itr, ITR))

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)
    print("rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

    return model, rmse, accuracy