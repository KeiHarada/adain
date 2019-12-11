# to run on server
import sys
sys.path.append("/home/harada/Documents/WorkSpace/adain")
sys.path.append("/home")

import pickle
import _pickle
import torch
import random
import math
import bz2
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from sklearn.metrics import mean_squared_error

# from my library
from source.model import ADAIN
from source.model import HARADA
from source.model import FNN
from source.utility import Color
from source.utility import MyDataset_ADAIN
from source.utility import MyDataset_HARADA
from source.utility import MyDataset_FNN
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

def makeDataset(cities, model_attribute, lstm_data_width, data_length=None):

    '''
    cities (list):
    :param model_attribute:
    :param lstm_data_width:
    :return:
    '''

    print("load data from the DB ... ", end="")

    '''
    station data
    '''
    for city in cities:
        station = pd.read_csv("database/station/station_{}.csv".format(city), dtype=object)

        if city == cities[0]:
            station_raw = station
        else:
            station_raw = pd.concat([station_raw, station], ignore_index=True)

    '''
    road data
    '''
    road_attribute = ["motorway", "trunk", "others"]
    dtype = {att: "float" for att in road_attribute}
    dtype["sid"] = "object"
    for city in cities:
        station = pd.read_csv("database/road/road_{}.csv".format(city), dtype=dtype)
        df = normalization(station[road_attribute])
        station = pd.concat([station.drop(road_attribute, axis=1), df], axis=1)

        if city == cities[0]:
            road_raw = station
        else:
            road_raw = pd.concat([road_raw, station], ignore_index=True)

    '''
    poi data
    '''
    poi_attribute =["Arts & Entertainment", "College & University", "Event",
                    "Food", "Nightlife Spot", "Outdoors & Recreation", "Professional & Other Places",
                    "Residence", "Shop & Service", "Travel & Transport"]

    dtype = {att: "float" for att in poi_attribute}
    dtype["sid"] = "object"
    for city in cities:
        station = pd.read_csv("database/poi/poi_{}.csv".format(city), dtype=dtype)
        df = normalization(station[poi_attribute])
        station = pd.concat([station.drop(poi_attribute, axis=1), df], axis=1)

        if city == cities[0]:
            poi_raw = station
        else:
            poi_raw = pd.concat([poi_raw, station], ignore_index=True)

    '''
    meteorology data
    '''
    meteorology_attribute = ["weather", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
    dtype = {att: "float" for att in meteorology_attribute}
    dtype["did"], dtype["time"] = "object", "object"
    for city in cities:
        station = pd.read_csv("database/meteorology/meteorology_{}.csv".format(city), dtype=dtype)
        meteorology_attribute = ["temperature", "pressure", "humidity", "wind_speed"]
        df = normalization(data_interpolate(station[meteorology_attribute]))
        station = pd.concat([station.drop(meteorology_attribute, axis=1), df], axis=1)

        df, columns = weather_onehot(station["weather"])
        station = pd.concat([station.drop(["weather"], axis=1), df], axis=1)
        meteorology_attribute += columns

        df, columns = winddirection_onehot(station["wind_direction"])
        station = pd.concat([station.drop(["wind_direction"], axis=1), df], axis=1)
        meteorology_attribute += columns

        if city == cities[0]:
            meteorology_raw = station
        else:
            meteorology_raw = pd.concat([meteorology_raw, station], ignore_index=True)

    '''
    aqi data
    '''
    aqi_attribute = ["pm25", "pm10", "no2", "co", "o3", "so2"]
    dtype = {att: "float" for att in aqi_attribute}
    dtype["sid"], dtype["time"] = "object", "object"
    for city in cities:
        # for label
        station_label = pd.read_csv("database/aqi/aqi_{}.csv".format(city), dtype=dtype)
        df = data_interpolate(station_label[[model_attribute]])
        station_label = pd.concat([station_label.drop(aqi_attribute, axis=1), df], axis=1)

        # for feature
        station_feature = pd.read_csv("database/aqi/aqi_{}.csv".format(city), dtype=dtype)
        df = normalization(data_interpolate(station_feature[[model_attribute]]))
        station_feature = pd.concat([station_feature.drop(aqi_attribute, axis=1), df], axis=1)

        if city == cities[0]:
            aqi_raw_label = station_label
            aqi_raw_feature = station_feature
        else:
            aqi_raw_label = pd.concat([aqi_raw_label, station_label], ignore_index=True)
            aqi_raw_feature = pd.concat([aqi_raw_feature, station_feature], ignore_index=True)

    print(Color.GREEN + "OK" + Color.END)
    print("make data ... ", end="")

    '''
    make dataset
    '''
    staticData, seqData_m, seqData_a, labelData = dict(), dict(), dict(), dict()
    for sid in list(station_raw["sid"]):

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
    with open("datatmp/inputDim.pkl", "wb") as fp:
        dc = {"local_static": len(poi_attribute)+len(road_attribute),
              "local_seq": len(meteorology_attribute),
              "others_static": len(poi_attribute)+len(road_attribute) + 2, # + distance and angle
              "others_seq": len(meteorology_attribute) + 1} # + aqi
        pickle.dump(dc, fp)

    with open("datatmp/stationData.pkl", "wb") as fp:
        pickle.dump(station_raw, fp)

    with open("datatmp/staticData.pkl", "wb") as fp:
        pickle.dump(staticData, fp)

    with open("datatmp/meteorologyData.pkl", "wb") as fp:
        pickle.dump(seqData_m, fp)

    with open("datatmp/aqiData.pkl", "wb") as fp:
        pickle.dump(seqData_a, fp)

    with open("datatmp/labelData.pkl", "wb") as fp:
        pickle.dump(labelData, fp)

    print(Color.GREEN + "OK" + Color.END)

def makeCityData(trainPath, testPath, savePath_train, savePath_test):

    # static data
    staticData = pickle.load(open("datatmp/staticData.pkl", "rb"))

    # station, district, city data
    stationData = pd.read_csv("rawdata/zheng2015/station.csv", dtype=object)
    districtData = pd.read_csv("rawdata/zheng2015/district.csv", dtype=object)
    cityData = pd.read_csv("rawdata/zheng2015/city.csv", dtype=object)

    # the number of data files
    trainNum = pickle.load(open("{}/fileNum.pkl".format(trainPath), "rb"))["train"]
    for idx in range(trainNum):
        selector = "/train_{}.pkl.bz2".format(str(idx).zfill(3))
        local_static, local_seq, others_static, others_seq, target = pickle.load(bz2.BZ2File(trainPath + selector, 'rb'))
        dataNum, local_static, others_static = len(target), local_static[0], others_static[0]

        # local's location
        tmp = [k for k, v in staticData.items() if v == local_static][0]
        tmp = list(stationData[stationData["station_id"] == tmp]["district_id"])[0]
        tmp = list(districtData[districtData["district_id"] == tmp]["city_id"])[0]
        cid_local = tmp
        local_lat = float(cityData[cityData["city_id"] == tmp]["latitude"])
        local_lon = float(cityData[cityData["city_id"] == tmp]["longitude"])

        # output vectors
        city_index = list()
        station_index = list()
        geoVect = list()
        staticVect = list()

        for others_static_i in others_static:
            tmp = others_static_i[:-2]
            tmp = [k for k, v in staticData.items() if v == tmp][0]
            tmp = list(stationData[stationData["station_id"] == tmp]["district_id"])[0]
            cid = list(districtData[districtData["district_id"] == tmp]["city_id"])[0]
            station_index.append(cid)

            if cid == cid_local:
                continue

            if cid not in city_index:
                city_index.append(cid)
                geoVect.append([])
                staticVect.append([])

            other_lat = float(cityData[cityData["city_id"] == cid]["latitude"])
            other_lon = float(cityData[cityData["city_id"] == cid]["longitude"])
            tmp = get_dist_angle(local_lat, local_lon, other_lat, other_lon)

            geoVect[city_index.index(cid)] = [tmp["distance"], tmp["azimuth1"]]
            staticVect[city_index.index(cid)].append(others_static_i)

        # normalization others' location
        geoVect = np.array(geoVect)
        minimum = geoVect.min(axis=0, keepdims=True)
        maximum = geoVect.max(axis=0, keepdims=True)
        geoVect = (geoVect - minimum) / (maximum - minimum)
        geoVect = list(map(lambda x: list(x), geoVect))

        # group sequence data
        seqVect = list()
        for others_seq_i in others_seq:
            tmp = [[] for i in range(len(city_index))]
            for j in range(len(others_seq_i)):
                cid = station_index[j]
                if cid == cid_local:
                    continue
                tmp[city_index.index(cid)].append(others_seq_i[j])
            seqVect.append(tmp)

        out_set = ([local_static]*dataNum, local_seq, [staticVect]*dataNum, seqVect, [geoVect]*dataNum, target)

        with bz2.BZ2File("{}/train_{}.pkl.bz2".format(savePath_train, str(idx).zfill(3)), 'wb', compresslevel=9) as fp:
            fp.write(pickle.dumps(out_set))
            print("* save train_{}.pkl.bz2".format(str(idx).zfill(3)))

    # the number of data files
    validNum = pickle.load(open("{}/fileNum.pkl".format(trainPath), "rb"))["valid"]
    for idx in range(validNum):
        selector = "/valid_{}.pkl.bz2".format(str(idx).zfill(3))
        local_static, local_seq, others_static, others_seq, target = pickle.load(bz2.BZ2File(trainPath + selector, 'rb'))
        dataNum, local_static, others_static = len(target), local_static[0], others_static[0]

        # local's location
        tmp = [k for k, v in staticData.items() if v == local_static][0]
        tmp = list(stationData[stationData["station_id"] == tmp]["district_id"])[0]
        tmp = list(districtData[districtData["district_id"] == tmp]["city_id"])[0]
        cid_local = tmp
        local_lat = float(cityData[cityData["city_id"] == tmp]["latitude"])
        local_lon = float(cityData[cityData["city_id"] == tmp]["longitude"])

        # output vectors
        city_index = list()
        station_index = list()
        geoVect = list()
        staticVect = list()

        for others_static_i in others_static:
            tmp = others_static_i[:-2]
            tmp = [k for k, v in staticData.items() if v == tmp][0]
            tmp = list(stationData[stationData["station_id"] == tmp]["district_id"])[0]
            cid = list(districtData[districtData["district_id"] == tmp]["city_id"])[0]
            station_index.append(cid)

            if cid == cid_local:
                continue

            if cid not in city_index:
                city_index.append(cid)
                geoVect.append([])
                staticVect.append([])

            other_lat = float(cityData[cityData["city_id"] == cid]["latitude"])
            other_lon = float(cityData[cityData["city_id"] == cid]["longitude"])
            tmp = get_dist_angle(local_lat, local_lon, other_lat, other_lon)

            geoVect[city_index.index(cid)] = [tmp["distance"], tmp["azimuth1"]]
            staticVect[city_index.index(cid)].append(others_static_i)

        # normalization others' location
        geoVect = np.array(geoVect)
        minimum = geoVect.min(axis=0, keepdims=True)
        maximum = geoVect.max(axis=0, keepdims=True)
        geoVect = (geoVect - minimum) / (maximum - minimum)
        geoVect = list(map(lambda x: list(x), geoVect))

        # group sequence data
        seqVect = list()
        for others_seq_i in others_seq:
            tmp = [[] for i in range(len(city_index))]
            for j in range(len(others_seq_i)):
                cid = station_index[j]
                if cid == cid_local:
                    continue
                tmp[city_index.index(cid)].append(others_seq_i[j])
            seqVect.append(tmp)

        out_set = ([local_static]*dataNum, local_seq, [staticVect]*dataNum, seqVect, [geoVect]*dataNum, target)

        with bz2.BZ2File("{}/valid_{}.pkl.bz2".format(savePath_train, str(idx).zfill(3)), 'wb', compresslevel=9) as fp:
            fp.write(pickle.dumps(out_set))
            print("* save valid_{}.pkl.bz2".format(str(idx).zfill(3)))

    # the number of data files
    testNum = pickle.load(open("{}/fileNum.pkl".format(testPath), "rb"))["test"]
    for idx in range(testNum):
        selector = "/test_{}.pkl.bz2".format(str(idx).zfill(3))
        local_static, local_seq, others_static, others_seq, target = pickle.load(bz2.BZ2File(testPath + selector, 'rb'))
        dataNum, local_static, others_static = len(target), local_static[0], others_static[0]

        # local's location
        tmp = [k for k, v in staticData.items() if v == local_static][0]
        tmp = list(stationData[stationData["station_id"] == tmp]["district_id"])[0]
        tmp = list(districtData[districtData["district_id"] == tmp]["city_id"])[0]
        local_lat = float(cityData[cityData["city_id"] == tmp]["latitude"])
        local_lon = float(cityData[cityData["city_id"] == tmp]["longitude"])

        # output vectors
        city_index = list()
        station_index = list()
        geoVect = list()
        staticVect = list()

        for others_static_i in others_static:
            tmp = others_static_i[:-2]
            tmp = [k for k, v in staticData.items() if v == tmp][0]
            tmp = list(stationData[stationData["station_id"] == tmp]["district_id"])[0]
            cid = list(districtData[districtData["district_id"] == tmp]["city_id"])[0]
            station_index.append(cid)

            if cid not in city_index:
                city_index.append(cid)
                geoVect.append([])
                staticVect.append([])

            other_lat = float(cityData[cityData["city_id"] == cid]["latitude"])
            other_lon = float(cityData[cityData["city_id"] == cid]["longitude"])
            tmp = get_dist_angle(local_lat, local_lon, other_lat, other_lon)

            geoVect[city_index.index(cid)] = [tmp["distance"], tmp["azimuth1"]]
            staticVect[city_index.index(cid)].append(others_static_i)

        # remove a city
        tmp = list(map(lambda x: len(x), staticVect))
        tmp = tmp.index(min(tmp))
        cid_removed = city_index[tmp]
        del staticVect[tmp], geoVect[tmp], city_index[tmp]

        # normalization others' location
        geoVect = np.array(geoVect)
        minimum = geoVect.min(axis=0, keepdims=True)
        maximum = geoVect.max(axis=0, keepdims=True)
        geoVect = (geoVect - minimum) / (maximum - minimum)
        geoVect = list(map(lambda x: list(x), geoVect))

        # group sequence data
        seqVect = list()
        for others_seq_i in others_seq:
            tmp = [[] for i in range(len(city_index))]
            for j in range(len(others_seq_i)):
                cid = station_index[j]
                if cid == cid_removed:
                    continue
                tmp[city_index.index(cid)].append(others_seq_i[j])
            seqVect.append(tmp)

        out_set = ([local_static]*dataNum, local_seq, [staticVect]*dataNum, seqVect, [geoVect]*dataNum, target)

        with bz2.BZ2File("{}/test_{}.pkl.bz2".format(savePath_test, str(idx).zfill(3)), 'wb', compresslevel=9) as fp:
            fp.write(pickle.dumps(out_set))
            print("* save test_{}.pkl.bz2".format(str(idx).zfill(3)))

    with open("{}/fileNum.pkl".format(savePath_test), "wb") as fp:
        pickle.dump({"test": testNum}, fp)
    with open("{}/fileNum.pkl".format(savePath_train), "wb") as fp:
        pickle.dump({"train": trainNum, "valid": validNum}, fp)

def makeTrainData(savePath, station_train):

    '''
    :param station_train): a list of station ids
    :return: featureData, labelData
    '''

    # raw data
    stationData = pickle.load(open("datatmp/stationData.pkl", "rb"))
    staticData = pickle.load(open("datatmp/staticData.pkl", "rb"))
    meteorologyData = pickle.load(open("datatmp/meteorologyData.pkl", "rb"))
    aqiData = pickle.load(open("datatmp/aqiData.pkl", "rb"))
    targetData = pickle.load(open("datatmp/labelData.pkl", "rb"))

    '''
    featureData_t = (local_static, local_seq, others_static, others_seq)_t
    labelData_t = target_t
    '''

    trainNum = math.floor(len(station_train)*0.9)

    tdx, vdx = 0, 0
    for station_local in station_train:

        # output
        out_local_static = list()
        out_local_seq = list()
        out_others_static = list()
        out_others_seq = list()
        out_target = list()

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

        out_set = (out_local_static, out_local_seq, out_others_static, out_others_seq, out_target)

        if tdx > trainNum-1:
            with bz2.BZ2File("{}/valid_{}.pkl.bz2".format(savePath, str(vdx).zfill(3)), 'wb', compresslevel=9) as fp:
                fp.write(pickle.dumps(out_set))
                print("* save valid_{}.pkl.bz2".format(str(vdx).zfill(3)))
            vdx += 1
        else:
            with bz2.BZ2File("{}/train_{}.pkl.bz2".format(savePath, str(tdx).zfill(3)), 'wb', compresslevel=9) as fp:
                fp.write(pickle.dumps(out_set))
                print("* save train_{}.pkl.bz2".format(str(tdx).zfill(3)))
            tdx += 1

        del out_local_seq, out_local_static, out_others_seq, out_others_static, out_set

    with open("{}/fileNum.pkl".format(savePath), "wb") as fp:
        pickle.dump({"train": tdx, "valid": vdx}, fp)

def makeTestData(savePath, station_test, station_train):
    '''
    :param station_train): a list of station ids
    :return: featureData, labelData
    '''

    # raw data
    stationData = pickle.load(open("datatmp/stationData.pkl", "rb"))
    staticData = pickle.load(open("datatmp/staticData.pkl", "rb"))
    meteorologyData = pickle.load(open("datatmp/meteorologyData.pkl", "rb"))
    aqiData = pickle.load(open("datatmp/aqiData.pkl", "rb"))
    targetData = pickle.load(open("datatmp/labelData.pkl", "rb"))

    '''
    featureData_t = (local_static, local_seq, others_static, others_seq)_t
    labelData_t = target_t
    '''

    tdx = 0
    for station_local in station_test:

        for station_removed in station_train:

            # output
            out_local_static = list()
            out_local_seq = list()
            out_others_static = list()
            out_others_seq = list()
            out_target = list()

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
            make datatmp
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

            out_set = (out_local_static, out_local_seq, out_others_static, out_others_seq, out_target)

            with bz2.BZ2File("{}/test_{}.pkl.bz2".format(savePath, str(tdx).zfill(3)), 'wb', compresslevel=9) as fp:
                fp.write(pickle.dumps(out_set))
                print("* save test_{}.pkl.bz2".format(str(tdx).zfill(3)))
            tdx += 1

            del out_local_seq, out_local_static, out_others_seq, out_others_static, out_set

    with open("{}/fileNum.pkl".format(savePath), "wb") as fp:
        pickle.dump({"test": tdx}, fp)

def makeTestData_sampled(savePath, station_test, station_train):
    '''
    :param station_train): a list of station ids
    :return: featureData, labelData
    '''

    # raw data
    stationData = pickle.load(open("datatmp/stationData.pkl", "rb"))
    staticData = pickle.load(open("datatmp/staticData.pkl", "rb"))
    meteorologyData = pickle.load(open("datatmp/meteorologyData.pkl", "rb"))
    aqiData = pickle.load(open("datatmp/aqiData.pkl", "rb"))
    targetData = pickle.load(open("datatmp/labelData.pkl", "rb"))

    '''
    featureData_t = (local_static, local_seq, others_static, others_seq)_t
    labelData_t = target_t
    '''

    tdx = 0
    for station_local in station_test:

        # output
        out_local_static = list()
        out_local_seq = list()
        out_others_static = list()
        out_others_seq = list()
        out_target = list()

        station_others = station_train.copy()
        station_remove = station_others[random.randint(0, len(station_others)-1)]
        station_others.remove(station_remove)
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
        make datatmp
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

        out_set = (out_local_static, out_local_seq, out_others_static, out_others_seq, out_target)

        with bz2.BZ2File("{}/test_{}.pkl.bz2".format(savePath, str(tdx).zfill(3)), 'wb', compresslevel=9) as fp:
            fp.write(pickle.dumps(out_set))
            print("* save test_{}.pkl.bz2".format(str(tdx).zfill(3)))
        tdx += 1

        del out_local_seq, out_local_static, out_others_seq, out_others_static, out_set

    with open("{}/fileNum.pkl".format(savePath), "wb") as fp:
        pickle.dump({"test": tdx}, fp)

def objective_ADAIN(trial):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyper parameters for tuning
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    # epochs = trial.suggest_discrete_uniform("epochs", 1, 5, 1)
    # lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    # wd = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    # hyper parameters for constance
    batch_size = 256
    epochs = 200
    lr = 0.001
    wd = 0.0005

    # input dimension
    inputDim = pickle.load(open("datatmp/inputDim.pkl", "rb"))

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

    # initialize the early stopping object
    patience = 50
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # log
    logs = list()

    # dataset path
    dataPath = pickle.load(open("tmp/trainPath.pkl", "rb"))

    # the number which the train/validation dataset was divivded into
    trainNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["train"]
    validNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["valid"]

    # start training
    for step in range(int(epochs)):

        epoch_loss = list()

        # train
        for idx in range(trainNum):

            selector = "/train_{}.pkl.bz2".format(str(idx).zfill(3))
            trainData = MyDataset_ADAIN(pickle.load(bz2.BZ2File(dataPath + selector, 'rb')))
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
        for idx in range(validNum):
            selector = "/valid_{}.pkl.bz2".format(str(idx).zfill(3))
            validData = MyDataset_ADAIN(pickle.load(bz2.BZ2File(dataPath + selector, 'rb')))
            rmse_i, accuracy_i = validate_ADAIN(model, validData)
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
    with open("tmp/{}_model.pkl".format(str(trial_num).zfill(4)), "wb") as pl:
        torch.save(model.state_dict(), pl)

    # save logs
    logs = pd.DataFrame(logs)
    with open("tmp/{}_log.pkl".format(str(trial_num).zfill(4)), "wb") as pl:
        pickle.dump(logs, pl)

    return rmse

def validate_ADAIN(model, validData):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for evaluation
    result = list()
    result_label = list()

    batch_size = 200

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

def evaluate_ADAIN(model_state_dict):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # input dimension
    inputDim = pickle.load(open("datatmp/inputDim.pkl", "rb"))

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
    result = list()
    result_label = list()

    # dataset path
    dataPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # the number which the test dataset was divided into
    testNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["test"]

    batch_size = 200
    iteration = 0

    for idx in range(testNum):

        selector = "/test_{}.pkl.bz2".format(str(idx).zfill(3))
        testData = MyDataset_ADAIN(pickle.load(bz2.BZ2File(dataPath + selector, 'rb')))
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
            print("\t|- iteration %d / %d" % (iteration, len(testData)*testNum))

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)
    print("rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

    return rmse, accuracy

def objective_HARADA(trial):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyper parameters for tuning
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    # epochs = trial.suggest_discrete_uniform("epochs", 1, 5, 1)
    # lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    # wd = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    # hyper parameters for constance
    batch_size = 256
    epochs = 1
    lr = 0.001
    wd = 0.0005

    # input dimension
    inputDim = pickle.load(open("datatmp/inputDim.pkl", "rb"))
    cityNum = 4
    stationNum = 5

    # model
    model = HARADA(inputDim_local_static=inputDim["local_static"],
                   inputDim_local_seq=inputDim["local_seq"],
                   inputDim_others_static=inputDim["others_static"],
                   inputDim_others_seq=inputDim["others_seq"],
                   cityNum=cityNum,
                   stationNum=stationNum)

    # GPU or CPU
    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # evaluation function
    criterion = nn.MSELoss()

    # initialize the early stopping object
    patience = 50
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # log
    logs = list()

    # dataset path
    dataPath = pickle.load(open("tmp/trainPath.pkl", "rb"))

    # the number which the train/validation dataset was divivded into
    trainNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["train"]
    validNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["valid"]

    # start training
    for step in range(int(epochs)):

        epoch_loss = list()

        # train
        for idx in range(trainNum):

            selector = "/train_{}.pkl.bz2".format(str(idx).zfill(3))
            trainData = MyDataset_HARADA(pickle.load(bz2.BZ2File(dataPath + selector, 'rb')))
            for batch_i in torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True):

                print("\t|- batch loss: ", end="")

                # initialize graduation
                optimizer.zero_grad()

                # batch data
                batch_local_static, \
                batch_local_seq, \
                batch_others_static, \
                batch_others_seq, \
                batch_others_city, \
                batch_target = batch_i

                # to GPU
                batch_local_static = batch_local_static.to(device)
                batch_local_seq = batch_local_seq.to(device)
                batch_others_static = batch_others_static.to(device)
                batch_others_seq = batch_others_seq.to(device)
                batch_others_city = batch_others_city.to(device)
                batch_target = batch_target.to(device)

                # predict
                pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city)

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
        for idx in range(validNum):
            selector = "/valid_{}.pkl.bz2".format(str(idx).zfill(3))
            validData = MyDataset_HARADA(pickle.load(bz2.BZ2File(dataPath + selector, 'rb')))
            rmse_i, accuracy_i = validate_ADAIN(model, validData)
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
    with open("tmp/{}_model.pkl".format(str(trial_num).zfill(4)), "wb") as pl:
        torch.save(model.state_dict(), pl)

    # save logs
    logs = pd.DataFrame(logs)
    with open("tmp/{}_log.pkl".format(str(trial_num).zfill(4)), "wb") as pl:
        pickle.dump(logs, pl)

    return rmse

def validate_HARADA(model, validData):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for evaluation
    result = list()
    result_label = list()

    batch_size = 200

    for batch_i in torch.utils.data.DataLoader(validData, batch_size=batch_size, shuffle=False):

        batch_local_static, \
        batch_local_seq, \
        batch_others_static, \
        batch_others_seq, \
        batch_others_city, \
        batch_target = batch_i

        # to GPU
        batch_local_static = batch_local_static.to(device)
        batch_local_seq = batch_local_seq.to(device)
        batch_others_static = batch_others_static.to(device)
        batch_others_seq = batch_others_seq.to(device)
        batch_others_city = batch_others_city.to(device)

        # predict
        pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city)
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

def evaluate_HARADA(model_state_dict):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # input dimension
    inputDim = pickle.load(open("datatmp/inputDim.pkl", "rb"))
    cityNum = 4
    stationNum = 5

    # model
    model = HARADA(inputDim_local_static=inputDim["local_static"],
                   inputDim_local_seq=inputDim["local_seq"],
                   inputDim_others_static=inputDim["others_static"],
                   inputDim_others_seq=inputDim["others_seq"],
                   cityNum=cityNum,
                   stationNum=stationNum)

    model.load_state_dict(model_state_dict)
    model = model.to(device)

    # evaluate mode
    model.eval()

    # for evaluation
    result = list()
    result_label = list()

    # dataset path
    dataPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # the number which the test dataset was divided into
    testNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["test"]

    batch_size = 200
    iteration = 0

    for idx in range(testNum):

        selector = "/test_{}.pkl.bz2".format(str(idx).zfill(3))
        testData = MyDataset_HARADA(pickle.load(bz2.BZ2File(dataPath + selector, 'rb')))
        for batch_i in torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False):

            batch_local_static, \
            batch_local_seq, \
            batch_others_static, \
            batch_others_seq, \
            batch_others_city, \
            batch_target = batch_i

            # to GPU
            batch_local_static = batch_local_static.to(device)
            batch_local_seq = batch_local_seq.to(device)
            batch_others_static = batch_others_static.to(device)
            batch_others_seq = batch_others_seq.to(device)
            batch_others_city = batch_others_city.to(device)

            # predict
            pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city)
            pred = pred.to("cpu")

            # evaluate
            pred = list(map(lambda x: x[0], pred.data.numpy()))
            batch_target = list(map(lambda x: x[0], batch_target.data.numpy()))
            result += pred
            result_label += batch_target

            iteration += len(batch_target)
            print("\t|- iteration %d / %d" % (iteration, len(testData)*testNum))

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)
    print("rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

    return rmse, accuracy

def objective_FNN(trial):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyper parameters for tuning
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    # epochs = trial.suggest_discrete_uniform("epochs", 1, 5, 1)
    # lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    # wd = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    # hyper parameters for constance
    batch_size = 1024
    epochs = 200
    lr = 0.001
    wd = 0.0005

    # dataset path
    dataPath = pickle.load(open("tmp/trainPath.pkl", "rb"))
    selector = "/train_000.pkl.bz2"

    # dataset
    local_static, local_seq, others_static, others_seq, target = pickle.load(bz2.BZ2File(dataPath + selector, 'rb'))

    dataset = list()
    inputDim = 0
    for i in range(len(target)):
        trainData_i = list()
        trainData_i += local_static[i]
        trainData_i += local_seq[i][-1]
        for j in range(len(others_static[i])):
            trainData_i += others_static[i][j]
            trainData_i += others_seq[i][j][-1]
        dataset.append(trainData_i)
        inputDim = len(trainData_i)

    # train and validation data
    trainNum = math.floor(len(dataset)*0.67)
    trainData = MyDataset_FNN(dataset[:trainNum], target[:trainNum])
    validData = MyDataset_FNN(dataset[trainNum:], target[trainNum:])
    del dataset

    # model
    model = FNN(inputDim=inputDim)

    # GPU or CPU
    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # evaluation function
    criterion = nn.MSELoss()

    # initialize the early stopping object
    patience = 50
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # log
    logs = list()

    # start training
    for step in range(int(epochs)):

        epoch_loss = list()

        for batch_i in torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True):

            print("\t|- batch loss: ", end="")

            # initialize graduation
            optimizer.zero_grad()

            # batch data
            batch_feature, batch_target = batch_i

            # to GPU
            batch_feature = batch_feature.to(device)
            batch_target = batch_target.to(device)

            # predict
            pred = model(batch_feature)

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
        model.eval()
        rmse, accuracy = validate_FNN(model, validData)
        model.train()

        # calculate validation loss
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
    with open("tmp/{}_model.pkl".format(str(trial_num).zfill(4)), "wb") as pl:
        torch.save(model.state_dict(), pl)

    # save logs
    logs = pd.DataFrame(logs)
    with open("tmp/{}_log.pkl".format(str(trial_num).zfill(4)), "wb") as pl:
        pickle.dump(logs, pl)

    return rmse

def validate_FNN(model, validData):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for evaluation
    result = list()
    result_label = list()

    batch_size = 2000

    for batch_i in torch.utils.data.DataLoader(validData, batch_size=batch_size, shuffle=False):

        # batch data
        batch_feature, batch_target = batch_i

        # to GPU
        batch_feature = batch_feature.to(device)

        # predict
        pred = model(batch_feature)
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

def evaluate_FNN(model_state_dict):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset path
    dataPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # the number which the test dataset was divided into
    testNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["test"]

    # dataset
    testData = list()
    labelData = list()
    inputDim = 0
    for idx in range(testNum):
        
        selector = "/test_{}.pkl.bz2".format(str(idx).zfill(3))
        local_static, local_seq, others_static, others_seq, target = pickle.load(bz2.BZ2File(dataPath + selector, 'rb'))
        for i in range(len(target)):
            testData_i = list()
            testData_i += local_static[i]
            testData_i += local_seq[i][-1]
            for j in range(len(others_static[i])):
                testData_i += others_static[i][j]
                testData_i += others_seq[i][j][-1]
            testData.append(testData_i)
            inputDim = len(testData_i)
        labelData += target
    testData = MyDataset_FNN(testData, labelData)

    # model
    model = FNN(inputDim=inputDim)
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    # evaluate mode
    model.eval()

    # for evaluation
    result = list()
    result_label = list()

    batch_size = 2000
    iteration = 0

    for batch_i in torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False):

        # batch data
        batch_feature, batch_target = batch_i

        # to GPU
        batch_feature = batch_feature.to(device)

        # predict
        pred = model(batch_feature)
        pred = pred.to("cpu")

        # evaluate
        pred = list(map(lambda x: x[0], pred.data.numpy()))
        batch_target = list(map(lambda x: x[0], batch_target.data.numpy()))
        result += pred
        result_label += batch_target

        iteration += len(batch_target)
        print("\t|- iteration %d / %d" % (iteration, len(testData)*testNum))

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)
    print("rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

    return rmse, accuracy

def evaluate_KNN(K):

    # aqi data
    aqiData = pickle.load(open("datatmp/labelData.pkl", "rb"))
    for k, v in aqiData.items():
        aqiData[k] = list(map(lambda x: x[0], v))

    # static data
    staticData = pickle.load(open("datatmp/staticData.pkl", "rb"))

    # station data
    stationData = pickle.load(open("datatmp/stationData.pkl", "rb"))

    # dataset path
    dataPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # the number which the test dataset was divided into
    testNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["test"]

    # for evaluation
    result = list()
    result_label = list()
    for idx in range(testNum):

        selector = "/test_{}.pkl.bz2".format(str(idx).zfill(3))
        dataset = pickle.load(bz2.BZ2File(dataPath + selector, 'rb'))
        local_static, others_static = dataset[0][0], dataset[2][0]

        # target station
        target_station = [k for k, v in staticData.items() if v == local_static][0]

        # source stations
        source_station = list()
        for staticData_i in others_static:
            staticData_i = staticData_i[:-2]
            source_station.append([k for k, v in staticData.items() if v == staticData_i][0])

        # calculate distance from the local station
        # local station
        lat_target = float(stationData[stationData["sid"] == target_station]["lat"])
        lon_target = float(stationData[stationData["sid"] == target_station]["lon"])

        # distance
        distance = dict()
        for source_station_i in source_station:
            lat_source = float(stationData[stationData["sid"] == source_station_i]["lat"])
            lon_source = float(stationData[stationData["sid"] == source_station_i]["lon"])
            distance[source_station_i] = get_dist_angle(lat1=lat_target, lon1=lon_target, lat2=lat_source, lon2=lon_source)["distance"]

        # get K nearest neighbors
        distance = sorted(distance.items(), key=lambda x: x[1])
        nearest = list(map(lambda x: x[0], distance[:K]))

        # agi data of source cities
        aqiData_source = list()
        for source_station_i in nearest:
            aqiData_source.append(aqiData[source_station_i])

        # aqi data of target city
        aqiData_target = aqiData[target_station]

        # evaluate
        aqiData_source = list(np.mean(np.array(aqiData_source), axis=0))
        result += aqiData_source
        result_label += aqiData_target

    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)

    return rmse, accuracy

def analysis_KNN(K):

    # aqi data
    aqiData = pickle.load(open("datatmp/labelData.pkl", "rb"))
    for k, v in aqiData.items():
        aqiData[k] = list(map(lambda x: x[0], v))

    # static data
    staticData = pickle.load(open("datatmp/staticData.pkl", "rb"))

    # station data
    stationData = pickle.load(open("datatmp/stationData.pkl", "rb"))

    # dataset path
    dataPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # the number which the test dataset was divided into
    testNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["test"]

    # evaluate
    for idx in range(testNum):

        selector = "/test_{}.pkl.bz2".format(str(idx).zfill(3))
        dataset = pickle.load(bz2.BZ2File(dataPath + selector, 'rb'))
        local_static, others_static = dataset[0][0], dataset[2][0]

        # target station
        target_station = [k for k, v in staticData.items() if v == local_static][0]

        # source stations
        source_station = list()
        for staticData_i in others_static:
            staticData_i = staticData_i[:-2]
            source_station.append([k for k, v in staticData.items() if v == staticData_i][0])

        # calculate distance from the local station
        # local station
        lat_target = float(stationData[stationData["sid"] == target_station]["lat"])
        lon_target = float(stationData[stationData["sid"] == target_station]["lon"])

        # distance
        distance = dict()
        for source_station_i in source_station:
            lat_source = float(stationData[stationData["sid"] == source_station_i]["lat"])
            lon_source = float(stationData[stationData["sid"] == source_station_i]["lon"])
            result = get_dist_angle(lat1=lat_target, lon1=lon_target, lat2=lat_source, lon2=lon_source)
            distance[source_station_i] = result["distance"]

        # get K nearest neighbors
        distance = sorted(distance.items(), key=lambda x: x[1])
        nearest = list(map(lambda x: x[0], distance[:K]))

        # agi data of source cities
        aqiData_source = list()
        for source_station_i in nearest:
            aqiData_source.append(aqiData[source_station_i])

        # aqi data of target city
        aqiData_target = aqiData[target_station]

        # evaluate
        aqiData_source_mean = list(np.mean(np.array(aqiData_source), axis=0))
        rmse = np.sqrt(mean_squared_error(aqiData_target, aqiData_source_mean))

        with open("result/K{}_idx{}.csv".format(str(K), str(idx).zfill(3)), "w") as outfile:
            outfile.write("RMSE={}\n".format(str(rmse)))
            outfile.write("{},mean,{}\n".format(",".join(nearest), target_station))
            for t in range(len(aqiData_target)):
                tmp = list()
                for s in range(len(nearest)):
                    tmp.append(str(aqiData_source[s][t]))
                tmp.append(str(aqiData_source_mean[t]))
                tmp.append(str(aqiData_target[t]))
                outfile.write("{}\n".format(",".join(tmp)))

def evaluate_LI():

    # aqi data
    aqiData = pickle.load(open("datatmp/labelData.pkl", "rb"))
    for k, v in aqiData.items():
        aqiData[k] = list(map(lambda x: x[0], v))

    # static data
    staticData = pickle.load(open("datatmp/staticData.pkl", "rb"))

    # station data
    stationData = pickle.load(open("datatmp/stationData.pkl", "rb"))

    # dataset path
    dataPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # the number which the test dataset was divided into
    testNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["test"]

    # for evaluation
    result = list()
    result_label = list()
    for idx in range(testNum):

        selector = "/test_{}.pkl.bz2".format(str(idx).zfill(3))
        dataset = pickle.load(bz2.BZ2File(dataPath + selector, 'rb'))
        local_static, others_static = dataset[0][0], dataset[2][0]

        # target station
        target_station = [k for k, v in staticData.items() if v == local_static][0]

        # source stations
        source_station = list()
        for staticData_i in others_static:
            staticData_i = staticData_i[:-2]
            source_station.append([k for k, v in staticData.items() if v == staticData_i][0])

        # calculate distance from the local station
        # local station
        lat_target = float(stationData[stationData["sid"] == target_station]["lat"])
        lon_target = float(stationData[stationData["sid"] == target_station]["lon"])

        # distance
        distance = dict()
        for source_station_i in source_station:
            lat_source = float(stationData[stationData["sid"] == source_station_i]["lat"])
            lon_source = float(stationData[stationData["sid"] == source_station_i]["lon"])
            distance[source_station_i] = 1 / get_dist_angle(lat1=lat_target, lon1=lon_target, lat2=lat_source, lon2=lon_source)["distance"]

        # calculate population
        dist_sum = sum(list(distance.values()))
        distance = {source_station_i: distance[source_station_i]/dist_sum for source_station_i in list(distance.keys())}

        # agi data of source cities
        aqiData_source = list()
        for source_station_i in source_station:
            aqiData_source.append(list(distance[source_station_i] * np.array(aqiData[source_station_i])))

        # aqi data of target city
        aqiData_target = aqiData[target_station]

        # evaluate
        aqiData_source = list(np.mean(np.array(aqiData_source), axis=0))
        result += aqiData_source
        result_label += aqiData_target

    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)

    return rmse, accuracy