# to run on server
import sys
sys.path.append("/home/harada/Documents/WorkSpace/adain")
sys.path.append("/home")

import pickle
import _pickle
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from sklearn.metrics import mean_squared_error

# from my library
from source.model import ADAIN
from source.utility import Color
from source.utility import MyDataset
from source.utility import get_dist_angle
from source.utility import calc_correct
from source.utility import get_aqi_series
from source.utility import get_meteorology_series
from source.utility import get_road_data
from source.utility import get_poi_data
from source.utility import normalization
from source.utility import data_interpolate
from source.utility import weather_onehot
from source.utility import winddirection_onehot
from source.utility import EarlyStopping
from source.utility import get_activation
from source.utility import get_optimizer

def makeDataset_mmd(source_city, target_city, data_length=None):

    '''
    make dataset
    source city
    '''
    # station data
    station_s = pd.read_csv("database/station/station_" + source_city + ".csv", dtype=object)

    # poi data
    poi_attribute =["Arts & Entertainment", "College & University", "Event",
                    "Food", "Nightlife Spot", "Outdoors & Recreation", "Professional & Other Places",
                    "Residence", "Shop & Service", "Travel & Transport"]
    dtype = {att: "float" for att in poi_attribute}
    dtype["sid"] = "object"

    poi = pd.read_csv("database/poi/poi_" + source_city + ".csv", dtype=dtype)
    df = normalization(poi[poi_attribute])
    poi = pd.concat([poi.drop(poi_attribute, axis=1), df], axis=1)

    # road network data
    road_attribute = ["motorway", "trunk", "others"]
    dtype = {att: "float" for att in road_attribute}
    dtype["sid"] = "object"

    road = pd.read_csv("database/road/road_" + source_city + ".csv", dtype=dtype)
    df = normalization(road[road_attribute])
    road = pd.concat([road.drop(road_attribute, axis=1), df], axis=1)

    # meteorological data
    meteorology_attribute = ["weather", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
    dtype = {att: "float" for att in meteorology_attribute}
    dtype["did"], dtype["time"] = "object", "object"

    mete = pd.read_csv("database/meteorology/meteorology_" + source_city + ".csv", dtype=dtype)
    meteorology_attribute = ["temperature", "pressure", "humidity", "wind_speed"]
    df = normalization(data_interpolate(mete[meteorology_attribute]))
    mete = pd.concat([mete.drop(meteorology_attribute, axis=1), df], axis=1)

    df, columns = weather_onehot(mete["weather"])
    mete = pd.concat([mete.drop(["weather"], axis=1), df], axis=1)
    meteorology_attribute += columns

    df, columns = winddirection_onehot(mete["wind_direction"])
    mete = pd.concat([mete.drop(["wind_direction"], axis=1), df], axis=1)
    meteorology_attribute += columns

    for sid in list(station_s["sid"]):

        did = list(station_s[station_s["sid"] == sid]["did"])[0]
        if data_length is None:
            mete_tmp = mete[mete["did"] == did]
        else:
            mete_tmp = mete[mete["did"] == did][:data_length]
        mete_tmp = mete_tmp.drop(["did"], axis=1).reset_index(drop=True)
        df = pd.DataFrame({"sid": [sid]*len(mete_tmp)})
        mete_tmp = pd.concat([df, mete_tmp], axis=1)

        if sid == list(station_s["sid"])[0]:
            meteorology = mete_tmp
        else:
            meteorology = pd.concat([meteorology, mete_tmp], axis=0, ignore_index=True)

    source_data = pd.merge(meteorology, poi, on="sid")
    source_data = pd.merge(source_data, road, on="sid")
    source_data = source_data.drop(["sid", "time"], axis=1)
    source_data = source_data.values

    '''
    make dataset
    target city
    '''
    # station data
    station_t = pd.read_csv("database/station/station_" + target_city + ".csv", dtype=object)

    # poi data
    poi_attribute =["Arts & Entertainment", "College & University", "Event",
                    "Food", "Nightlife Spot", "Outdoors & Recreation", "Professional & Other Places",
                    "Residence", "Shop & Service", "Travel & Transport"]
    dtype = {att: "float" for att in poi_attribute}
    dtype["sid"] = "object"

    poi = pd.read_csv("database/poi/poi_" + target_city + ".csv", dtype=dtype)
    df = normalization(poi[poi_attribute])
    poi = pd.concat([poi.drop(poi_attribute, axis=1), df], axis=1)

    # road network data
    road_attribute = ["motorway", "trunk", "others"]
    dtype = {att: "float" for att in road_attribute}
    dtype["sid"] = "object"

    road = pd.read_csv("database/road/road_" + target_city + ".csv", dtype=dtype)
    df = normalization(road[road_attribute])
    road = pd.concat([road.drop(road_attribute, axis=1), df], axis=1)

    # meteorological data
    meteorology_attribute = ["weather", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
    dtype = {att: "float" for att in meteorology_attribute}
    dtype["did"], dtype["time"] = "object", "object"

    mete = pd.read_csv("database/meteorology/meteorology_" + target_city + ".csv", dtype=dtype)
    meteorology_attribute = ["temperature", "pressure", "humidity", "wind_speed"]
    df = normalization(data_interpolate(mete[meteorology_attribute]))
    mete = pd.concat([mete.drop(meteorology_attribute, axis=1), df], axis=1)

    df, columns = weather_onehot(mete["weather"])
    mete = pd.concat([mete.drop(["weather"], axis=1), df], axis=1)
    meteorology_attribute += columns

    df, columns = winddirection_onehot(mete["wind_direction"])
    mete = pd.concat([mete.drop(["wind_direction"], axis=1), df], axis=1)
    meteorology_attribute += columns

    for sid in list(station_t["sid"]):

        did = list(station_t[station_t["sid"] == sid]["did"])[0]
        if data_length is None:
            mete_tmp = mete[mete["did"] == did]
        else:
            mete_tmp = mete[mete["did"] == did][:data_length]
        mete_tmp = mete_tmp.drop(["did"], axis=1).reset_index(drop=True)
        df = pd.DataFrame({"sid": [sid]*len(mete_tmp)})
        mete_tmp = pd.concat([df, mete_tmp], axis=1)

        if sid == list(station_t["sid"])[0]:
            meteorology = mete_tmp
        else:
            meteorology = pd.concat([meteorology, mete_tmp], axis=0, ignore_index=True)

    target_data = pd.merge(meteorology, poi, on="sid")
    target_data = pd.merge(target_data, road, on="sid")
    target_data = target_data.drop(["sid", "time"], axis=1)
    target_data = target_data.values

    # return as a type of numpy
    return source_data, target_data


def makeDataset_multi(cities, model_attribute, lstm_data_width, data_length=None):
    '''
    cities (list):
    :param model_attribute:
    :param lstm_data_width:
    :return:
    '''

    print("dataset ... ", end="")

    '''
    station data
    '''
    for city in cities:
        target = pd.read_csv("database/station/station_" + city + ".csv", dtype=object)
        with open("dataset/station_"+city+".pickle", "wb") as pl:
            pickle.dump(list(target["sid"]), pl)

        if city == cities[0]:
            station_raw = target
        else:
            station_raw = pd.concat([station_raw, target], ignore_index=True)

    station_all = list(station_raw["sid"])

    '''
    road data
    '''
    road_attribute = ["motorway", "trunk", "others"]
    dtype = {att: "float" for att in road_attribute}
    dtype["sid"] = "object"

    for city in cities:
        target = pd.read_csv("database/road/road_" + city + ".csv", dtype=dtype)
        df = normalization(target[road_attribute])
        target = pd.concat([target.drop(road_attribute, axis=1), df], axis=1)

        if city == cities[0]:
            road_raw = target
        else:
            road_raw = pd.concat([road_raw, target], ignore_index=True)

    '''
    poi data
    '''
    poi_attribute =["Arts & Entertainment", "College & University", "Event",
                    "Food", "Nightlife Spot", "Outdoors & Recreation", "Professional & Other Places",
                    "Residence", "Shop & Service", "Travel & Transport"]

    dtype = {att: "float" for att in poi_attribute}
    dtype["sid"] = "object"

    for city in cities:
        target = pd.read_csv("database/poi/poi_" + city + ".csv", dtype=dtype)
        df = normalization(target[poi_attribute])
        target = pd.concat([target.drop(poi_attribute, axis=1), df], axis=1)

        if city == cities[0]:
            poi_raw = target
        else:
            poi_raw = pd.concat([poi_raw, target], ignore_index=True)

    '''
    meteorology data
    '''
    meteorology_attribute = ["weather", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
    dtype = {att: "float" for att in meteorology_attribute}
    dtype["did"], dtype["time"] = "object", "object"

    for city in cities:
        target = pd.read_csv("database/meteorology/meteorology_" + city + ".csv", dtype=dtype)
        meteorology_attribute = ["temperature", "pressure", "humidity", "wind_speed"]
        df = normalization(data_interpolate(target[meteorology_attribute]))
        target = pd.concat([target.drop(meteorology_attribute, axis=1), df], axis=1)

        df, columns = weather_onehot(target["weather"])
        target = pd.concat([target.drop(["weather"], axis=1), df], axis=1)
        meteorology_attribute += columns

        df, columns = winddirection_onehot(target["wind_direction"])
        target = pd.concat([target.drop(["wind_direction"], axis=1), df], axis=1)
        meteorology_attribute += columns

        if city == cities[0]:
            meteorology_raw = target
        else:
            meteorology_raw = pd.concat([meteorology_raw, target], ignore_index=True)

    '''
    aqi data
    '''
    aqi_attribute = ["pm25", "pm10", "no2", "co", "o3", "so2"]
    dtype = {att: "float" for att in aqi_attribute}
    dtype["sid"], dtype["time"] = "object", "object"

    for city in cities:
        # for label
        target_label = pd.read_csv("database/aqi/aqi_" + city + ".csv", dtype=dtype)
        df = data_interpolate(target_label[[model_attribute]])
        target_label = pd.concat([target_label.drop(aqi_attribute, axis=1), df], axis=1)

        # for feature
        target_feature = pd.read_csv("database/aqi/aqi_" + city + ".csv", dtype=dtype)
        df = normalization(data_interpolate(target_feature[[model_attribute]]))
        target_feature = pd.concat([target_feature.drop(aqi_attribute, axis=1), df], axis=1)

        if city == cities[0]:
            aqi_raw_label = target_label
            aqi_raw_feature = target_feature
        else:
            aqi_raw_label = pd.concat([aqi_raw_label, target_label], ignore_index=True)
            aqi_raw_feature = pd.concat([aqi_raw_feature, target_feature], ignore_index=True)

    '''
    make dataset
    '''
    staticData, seqData_m, seqData_a, labelData = dict(), dict(), dict(), dict()
    for sid in station_all:

        '''
        static data
            * poi
            * road network data

        format
            X = [poi attributes, road attributes]
        '''
        recode = [float(get_poi_data(data=poi_raw, sid=sid, attribute=att)) for att in poi_attribute]
        staticData[sid] = recode
        recode = [float(get_road_data(data=road_raw, sid=sid, attribute=att)) for att in road_attribute]
        staticData[sid] += recode

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
        aqi_data2 = get_aqi_series(data=aqi_raw_feature, sid=sid, attribute=model_attribute)

        if data_length is None:
            data_length = len(meteorology_data[meteorology_attribute[0]])

        seqData_m[sid] = []
        seqData_a[sid] = []
        terminal = data_length
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
                recode_a_p.append([aqi_data2[t]])
            seqData_m[sid].append(recode_m_p)
            seqData_a[sid].append(recode_a_p)
            start += 1
            end += 1

        '''
        label data
            * aqi data

        format
            aqi = "pm25" or  "pm10" or "no2" or "co" or "o3" or "so2"
            X = [ [aqi_p], [aqi_p+1], ..., [aqi_n] ]
        '''
        aqi_data1 = get_aqi_series(data=aqi_raw_label, sid=sid, attribute=model_attribute)
        labelData[sid] = []
        for t in range(lstm_data_width - 1, data_length):
            labelData[sid].append([aqi_data1[t]])

    # saving
    with open("dataset/dataDim.pickle", "wb") as pl:
        dc = {"static": len(poi_attribute)+len(road_attribute),
              "sequence": len(meteorology_attribute)}
        pickle.dump(dc, pl)

    with open("dataset/stationData.pickle", "wb") as pl:
        pickle.dump(station_raw, pl)

    with open("dataset/staticData.pickle", "wb") as pl:
        pickle.dump(staticData, pl)

    with open("dataset/meteorologyData.pickle", "wb") as pl:
        pickle.dump(seqData_m, pl)

    with open("dataset/aqiData.pickle", "wb") as pl:
        pickle.dump(seqData_a, pl)

    with open("dataset/labelData.pickle", "wb") as pl:
        pickle.dump(labelData, pl)

    print(Color.GREEN + "OK" + Color.END)

def makeDataset_single(city_name, model_attribute, lstm_data_width, data_length=None):
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
    poi data                                                                                          
    '''
    poi_attribute =["Arts & Entertainment", "College & University", "Event",
                    "Food", "Nightlife Spot", "Outdoors & Recreation", "Professional & Other Places",
                    "Residence", "Shop & Service", "Travel & Transport"]

    dtype = {att: "float" for att in poi_attribute}
    dtype["sid"] = "object"

    poi_raw = pd.read_csv("database/poi/poi_" + city_name + ".csv", dtype=dtype)
    df = normalization(poi_raw[poi_attribute])
    poi_raw = pd.concat([poi_raw.drop(poi_attribute, axis=1), df], axis=1)

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

    # for label
    aqi_raw_label = pd.read_csv("database/aqi/aqi_" + city_name + ".csv", dtype=dtype)
    df = data_interpolate(aqi_raw_label[[model_attribute]])
    aqi_raw_label = pd.concat([aqi_raw_label.drop(aqi_attribute, axis=1), df], axis=1)

    # for feature
    aqi_raw_feature = pd.read_csv("database/aqi/aqi_" + city_name + ".csv", dtype=dtype)
    df = normalization(data_interpolate(aqi_raw_feature[[model_attribute]]))
    aqi_raw_feature = pd.concat([aqi_raw_feature.drop(aqi_attribute, axis=1), df], axis=1)

    '''
    make dataset
    '''
    staticData, seqData_m, seqData_a, labelData = dict(), dict(), dict(), dict()
    for sid in station_all:

        '''
        static data
            * poi
            * road network

        format
            X = [motorway, trunk, others]
        '''
        recode = [float(get_poi_data(data=poi_raw, sid=sid, attribute=att)) for att in poi_attribute]
        staticData[sid] = recode
        recode = [float(get_road_data(data=road_raw, sid=sid, attribute=att)) for att in road_attribute]
        staticData[sid] += recode

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
        aqi_data2 = get_aqi_series(data=aqi_raw_feature, sid=sid, attribute=model_attribute)

        if data_length is None:
            data_length = len(meteorology_data[meteorology_attribute[0]])

        seqData_m[sid] = []
        seqData_a[sid] = []
        terminal = data_length
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
                recode_a_p.append([aqi_data2[t]])
            seqData_m[sid].append(recode_m_p)
            seqData_a[sid].append(recode_a_p)
            start += 1
            end += 1

        '''
        label data
            * aqi data

        format
            aqi = "pm25" or  "pm10" or "no2" or "co" or "o3" or "so2"
            X = [ [aqi_p], [aqi_p+1], ..., [aqi_n] ]
        '''
        aqi_data1 = get_aqi_series(data=aqi_raw_label, sid=sid, attribute=model_attribute)
        labelData[sid] = []
        for t in range(lstm_data_width - 1, data_length):
            labelData[sid].append([aqi_data1[t]])

    # saving
    with open("dataset/dataDim.pickle", "wb") as pl:
        dc = {"static": len(poi_attribute)+len(road_attribute),
              "sequence": len(meteorology_attribute)}
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

    with open("dataset/labelData.pickle", "wb") as pl:
        pickle.dump(labelData, pl)

    print(Color.GREEN + "OK" + Color.END)

def makeTestBatch(divided, batch_length):

    '''
    :param divided:
    :param batch_length:
    :return: a list of batch
    '''

    # input
    local_static, local_seq, others_static, others_seq, target = divided

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

def _makeRandomBatch(divided, batch_length, batch_size):

    '''
    :param divided: a set of (local_static, local_seq, others_static, others_seq, target)
    :param batch_length:
    :param batch_size:
    :return: a list of batches
    '''

    # input
    local_static, local_seq, others_static, others_seq, target = divided

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
        print(batch[i][4])

    return batch

def makeRandomBatch(divided, batch_length, batch_size):
    '''
    :param divided: a set of (local_static, local_seq, others_static, others_seq, target)
    :param batch_length:
    :param batch_size:
    :return: a list of batches
    '''

    # input
    local_static, local_seq, others_static, others_seq, target = divided

    # output
    batch = list()
    '''
    a batch = (batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target)
    '''
    for i in range(batch_length):
        batch_local_static = list()
        batch_local_seq = list()
        batch_others_static = list()
        batch_others_seq = list()
        batch_target = list()

        for j in range(batch_size):
            idx = np.random.randint(0, len(target) -1)
            batch_local_static.append(local_static[0])
            batch_local_seq.append(local_seq[idx])
            batch_target.append(target[idx])

            for k in range(len(others_static)):

                if j == 0:
                    batch_others_static.append([others_static[k][0]])
                    batch_others_seq.append([others_seq[k][idx]])
                else:
                    batch_others_static[k].append(others_static[k][0])
                    batch_others_seq[k].append(others_seq[k][idx])

        batch.append((batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target))

    return batch

def makeTrainData(station_train):

    '''
    :param station_train): a list of station ids
    :return: featureData, labelData
    '''

    # raw data
    stationData = pickle.load(open("dataset/stationData.pickle", "rb"))
    staticData = pickle.load(open("dataset/staticData.pickle", "rb"))
    meteorologyData = pickle.load(open("dataset/meteorologyData.pickle", "rb"))
    aqiData = pickle.load(open("dataset/aqiData.pickle", "rb"))
    targetData = pickle.load(open("dataset/labelData.pickle", "rb"))

    # output
    out_local_static = list()
    out_local_seq = list()
    out_others_static = list()
    out_others_seq = list()
    out_target = list()

    '''
    featureData_t = (local_static, local_seq, others_static, others_seq)_t
    labelData_t = target_t
    '''
    for station_local in station_train:

        station_others = station_train.copy()
        station_others.remove(station_local)

        '''
        calculate distance and angle of other stations from local stations
        '''
        # lat, lon of local station
        lat_local = float(stationData[stationData["sid"] == station_local]["lat"])
        lon_local = float(stationData[stationData["sid"] == station_local]["lon"])

        # distance and angle
        distance = list()
        angle = list()
        for sid in station_others:
            lat = float(stationData[stationData["sid"] == sid]["lat"])
            lon = float(stationData[stationData["sid"] == sid]["lon"])
            result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
            distance.append(result["distance"])
            angle.append(result["azimuth1"])

        # normalization

        if len(distance) == 1:
            distance = [1.0]
            angle = [1.0]
        else:
            maximum = max(distance)
            minimum = min(distance)
            distance = list(map(lambda x: (x - minimum) / (maximum - minimum), distance))
            maximum = max(angle)
            minimum = min(angle)
            angle = list(map(lambda x: (x - minimum) / (maximum - minimum), angle))

        # add
        others_static = list()
        idx = 0
        for sid in station_others:
            others_static.append(staticData[sid] + [distance[idx], angle[idx]])
            idx += 1

        '''
        concut meteorological data with aqi data of seqData of others
        '''
        seqData_others = dict()
        for sid in station_others:
            m = _pickle.loads(_pickle.dumps(meteorologyData[sid], -1))  # _pickleを使った高速コピー
            a = _pickle.loads(_pickle.dumps(aqiData[sid], -1))  # _pickleを使った高速コピー
            for i in range(len(m)):
                for j in range(len(m[i])):
                    m[i][j] += a[i][j]
            seqData_others[sid] = m

        '''
        local data and target data
        '''
        local_static = staticData[station_local]
        local_seq = meteorologyData[station_local]
        target = targetData[station_local]

        '''
        make dataset
        '''
        for t in range(len(target)):

            others_seq = list()
            for sid in station_others:
                others_seq.append(seqData_others[sid][t])

            out_local_static.append(local_static)
            out_local_seq.append(local_seq[t])
            out_others_static.append(others_static)
            out_others_seq.append(others_seq)
            out_target.append(target[t])

    return (out_local_static, out_local_seq, out_others_static, out_others_seq, out_target)

def makeTestData(station_test, station_train):

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
    targetData = pickle.load(open("dataset/labelData.pickle", "rb"))

    # output
    out_local_static = list()
    out_local_seq = list()
    out_others_static = list()
    out_others_seq = list()
    out_target = list()

    '''
    featureData_t = (local_static, local_seq, others_static, others_seq)_t
    labelData_t = target_t
    '''
    for station_local in station_test:

        for station_removed in station_train:

            station_others = station_train.copy()
            station_others.remove(station_removed)
            '''
            calculate distance and angle of other stations from local stations
            '''
            # lat, lon of local station
            lat_local = float(stationData[stationData["sid"] == station_local]["lat"])
            lon_local = float(stationData[stationData["sid"] == station_local]["lon"])

            # distance and angle
            distance = list()
            angle = list()
            for sid in station_others:
                lat = float(stationData[stationData["sid"] == sid]["lat"])
                lon = float(stationData[stationData["sid"] == sid]["lon"])
                result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
                distance.append(result["distance"])
                angle.append(result["azimuth1"])

            # normalization
            if len(distance) == 1:
                distance = [1.0]
                angle = [1.0]
            else:
                maximum, minimum = max(distance), min(distance)
                distance = list(map(lambda x: (x - minimum) / (maximum - minimum), distance))
                maximum, minimum = max(angle), min(angle)
                angle = list(map(lambda x: (x - minimum) / (maximum - minimum), angle))

            # add
            others_static = list()
            idx = 0
            for sid in station_others:
                others_static.append(staticData[sid] + [distance[idx], angle[idx]])
                idx += 1

            '''
            concut meteorological data with aqi data of seqData of others
            '''
            seqData_others = dict()
            for sid in station_others:
                m = _pickle.loads(_pickle.dumps(meteorologyData[sid], -1))  # _pickleを使った高速コピー
                a = _pickle.loads(_pickle.dumps(aqiData[sid], -1))  # _pickleを使った高速コピー
                for i in range(len(m)):
                    for j in range(len(m[i])):
                        m[i][j] += a[i][j]
                seqData_others[sid] = m

            '''
            local data and target data
            '''
            local_static = staticData[station_local]
            local_seq = meteorologyData[station_local]
            target = targetData[station_local]

            '''
            make dataset
            '''
            for t in range(len(target)):

                others_seq = list()
                for sid in station_others:
                    others_seq.append(seqData_others[sid][t])

                out_local_static.append(local_static)
                out_local_seq.append(local_seq[t])
                out_others_static.append(others_static)
                out_others_seq.append(others_seq)
                out_target.append(target[t])

    return (out_local_static, out_local_seq, out_others_static, out_others_seq, out_target)

def objective(trial):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyper parameters for tuning
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    # epochs = trial.suggest_discrete_uniform("epochs", 1, 5, 1)
    # lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    # wd = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    # hyper parameters for constance
    batch_size = 1024
    epochs = 1
    lr = 0.001
    wd = 0.0005

    # input dimension
    inputDim = pickle.load(open("model/inputDim.pickle", "rb"))

    # model
    model = ADAIN(inputDim_local_static=inputDim["local_static"],
                  inputDim_local_seq=inputDim["local_seq"],
                  inputDim_others_static=inputDim["others_static"],
                  inputDim_others_seq=inputDim["others_seq"])

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
    patience = 50
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # log
    logs = []

    # divide train dataset to save memory usage
    divide_num = 5

    # make dataset
    print("data loading ...", end="")
    idx = 0
    for station_train_i in list(np.array_split(station_train, divide_num)):
        # train data
        with open("tmp/train_{}.pickle".format(str(idx).zfill(2)), "wb") as pl:
            pickle.dump(MyDataset(makeTrainData(list(station_train_i))), pl)
        # validation data
        with open("tmp/valid_{}.pickle".format(str(idx).zfill(2)), "wb") as pl:
            pickle.dump(MyDataset(makeTestData(station_valid, list(station_train_i))), pl)
        idx += 1
    print(Color.GREEN + "OK" + Color.END)

    # start training
    for step in range(int(epochs)):

        epoch_loss = list()

        # train
        for idx in range(divide_num):

            trainData = pickle.load(open("tmp/train_{}.pickle".format(str(idx).zfill(2)), "rb"))

            for batch_i in torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True):

                print("\t|- batch loss: ", end="")

                # initialize graduation
                optimizer.zero_grad()

                # batch data
                batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target = batch_i

                # to GPU
                batch_local_static = batch_local_static.to(device)
                batch_local_seq = batch_local_seq.to(device)
                batch_others_static = batch_others_static.to(device)
                batch_others_seq = batch_others_seq.to(device)
                batch_target = batch_target.to(device)

                # predict
                pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq)

                # calculate loss, back-propagate loss, and step optimizer
                loss = criterion(pred, batch_target)
                loss.backward()
                optimizer.step()

                # print a batch loss as RMSE
                batch_loss = np.sqrt(loss.item())
                print("%.10f" % (batch_loss))

                # append batch loss to the list to calculate epoch loss
                epoch_loss.append(batch_loss)

        epoch_loss = np.average(epoch_loss)
        print("\t\t|- epoch %d loss: %.10f" % (step + 1, epoch_loss))

        # validate
        print("\t\t|- validation : ", end="")
        rmse = list()
        accuracy = list()
        model.eval()
        for idx in range(divide_num):
            validData = pickle.load(open("tmp/valid_{}.pickle".format(str(idx).zfill(2)), "rb"))
            rmse_i, accuracy_i = validate(model, validData)
            rmse.append(rmse_i)
            accuracy.append(accuracy_i)
        model.train()

        # calculate validation loss
        rmse = np.average(rmse)
        accuracy = np.average(accuracy)
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
    rmse = early_stopping.val_loss_min

    # save model
    trial_num = trial.number
    with open("tmp/" + str(trial_num).zfill(4) + "_model.pickle", "wb") as pl:
        torch.save(model.state_dict(), pl)

    # save logs
    logs = pd.DataFrame(logs)
    with open("tmp/" + str(trial_num).zfill(4) + "_log.pickle", "wb") as pl:
        pickle.dump(logs, pl)

    return rmse

def validate(model, validData):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for evaluation
    result = []
    result_label = []

    batch_size = 2000

    for batch_i in torch.utils.data.DataLoader(validData, batch_size=batch_size, shuffle=False):

        batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target = batch_i

        # to GPU
        batch_local_static = batch_local_static.to(device)
        batch_local_seq = batch_local_seq.to(device)
        batch_others_static = batch_others_static.to(device)
        batch_others_seq = batch_others_seq.to(device)

        # predict
        pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq)
        pred = pred.to("cpu")

        # evaluate
        pred = list(map(lambda x: x[0], pred.data.numpy()))
        batch_target = list(map(lambda x: x[0], batch_target.data.numpy()))
        result += pred
        result_label += batch_target

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)

    return rmse, accuracy

def evaluate(model_state_dict, station_train, station_test):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # input dimension
    inputDim = pickle.load(open("model/inputDim.pickle", "rb"))

    # model
    model = ADAIN(inputDim_local_static=inputDim["local_static"],
                  inputDim_local_seq=inputDim["local_seq"],
                  inputDim_others_static=inputDim["others_static"],
                  inputDim_others_seq=inputDim["others_seq"])

    model.load_state_dict(model_state_dict)
    model = model.to(device)

    # evaluate mode
    model.eval()

    # for evaluation
    result = []
    result_label = []

    # divide dataset to save memory usage
    divide_num = 5

    print("data loading ...", end="")
    idx = 0
    for station_train_i in list(np.array_split(station_train, divide_num)):
        with open("tmp/test_{}.pickle".format(str(idx).zfill(2)), "wb") as pl:
            pickle.dump(MyDataset(makeTestData(station_test, list(station_train_i))), pl)
        idx += 1
    print(Color.GREEN + "OK" + Color.END)

    batch_size = 2000
    iteration = 0

    for idx in range(divide_num):

        testData = pickle.load(open("tmp/test_{}.pickle".format(str(idx).zfill(2)), "rb"))

        for batch_i in torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False):

            batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target = batch_i

            # to GPU
            batch_local_static = batch_local_static.to(device)
            batch_local_seq = batch_local_seq.to(device)
            batch_others_static = batch_others_static.to(device)
            batch_others_seq = batch_others_seq.to(device)

            # predict
            pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq)
            pred = pred.to("cpu")

            # evaluate
            pred = list(map(lambda x: x[0], pred.data.numpy()))
            batch_target = list(map(lambda x: x[0], batch_target.data.numpy()))
            result += pred
            result_label += batch_target

            iteration += len(batch_target)
            print("\t|- iteration %d / %d" % (iteration, len(testData)))

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)
    print("rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

    return rmse, accuracy

def re_evaluate(model_state_dict, station_train, station_test, loop, city):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # input dimension
    inputDim = pickle.load(open("model/inputDim.pickle", "rb"))

    # model
    model = ADAIN(inputDim_local_static=inputDim["local_static"],
                  inputDim_local_seq=inputDim["local_seq"],
                  inputDim_others_static=inputDim["others_static"],
                  inputDim_others_seq=inputDim["others_seq"])

    model.load_state_dict(model_state_dict)
    model = model.to(device)

    # evaluate mode
    model.eval()

    # for evaluation
    result = []
    result_label = []

    # load data
    print("data loading ....", end="")
    testData = MyDataset(makeTestData(station_test, station_train))
    print(Color.GREEN + "OK" + Color.END)

    batch_size = 5000
    iteration = 0
    for batch_i in torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False):

        batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target = batch_i

        # to GPU
        batch_local_static = batch_local_static.to(device)
        batch_local_seq = batch_local_seq.to(device)
        batch_others_static = batch_others_static.to(device)
        batch_others_seq = batch_others_seq.to(device)

        # predict
        pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq)
        pred = pred.to("cpu")

        # evaluate
        pred = list(map(lambda x: x[0], pred.data.numpy()))
        batch_target = list(map(lambda x: x[0], batch_target.data.numpy()))
        result += pred
        result_label += batch_target

        iteration += len(batch_target)
        print("\t|- iteration %d / %d" % (iteration, len(testData)))

    with open("tmp/inferred_{}_{}.csv".format(str(loop).zfill(2), city), "w") as outfile:
        outfile.write("y_inf,y_label\n")
        for idx in range(len(result)):
            outfile.write("{},{}\n".format(str(result[idx]), str(result_label[idx])))

    with open("tmp/inferred_{}_{}_inf.pickle".format(str(loop).zfill(2), city), "wb") as pl:
        pickle.dump(result, pl)
    
    with open("tmp/inferred_{}_{}_label.pickle".format(str(loop).zfill(2), city), "wb") as pl:
        pickle.dump(result_label, pl)

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)
    print("rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

    return rmse, accuracy