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
from geomloss import SamplesLoss
from torch import optim
from sklearn.metrics import mean_squared_error

# from my library
from source.model import ADAIN
from source.model import HARADA
from source.model import _HARADA
from source.model import FNN
from source.utility import Color
from source.utility import MyDataset_ADAIN
from source.utility import MyDataset_HARADA
from source.utility import MyDataset_MMD
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

def makeCityData(dataPath_train, savePath_train, dataPath_test, savePath_test):

    # raw data
    stationRaw = pd.read_csv("rawdata/zheng2015/station.csv", dtype=object)
    districtRaw = pd.read_csv("rawdata/zheng2015/district.csv", dtype=object)
    cityRaw = pd.read_csv("rawdata/zheng2015/city.csv", dtype=object)

    # dataset
    stationData = pickle.load(open("datatmp/stationData.pkl", "rb"))
    staticData = pickle.load(open("datatmp/staticData.pkl", "rb"))
    meteorologyData = pickle.load(open("datatmp/meteorologyData.pkl", "rb"))
    aqiData = pickle.load(open("datatmp/aqiData.pkl", "rb"))
    targetData = pickle.load(open("datatmp/labelData.pkl", "rb"))

    # station train
    tmp = pickle.load(bz2.BZ2File("{}/train_000.pkl.bz2".format(dataPath_train), 'rb'))
    local, tmp = tmp[0][0], tmp[2][0]
    local = [k for k, v in staticData.items() if v == local][0]
    others = list()
    for static in tmp:
        static = static[:-2]
        others.append([k for k, v in staticData.items() if v == static][0])

    station_train = dict()
    for sid in [local] + others:
        did = list(stationRaw[stationRaw["station_id"] == sid]["district_id"])[0]
        cid = list(districtRaw[districtRaw["district_id"] == did]["city_id"])[0]
        if cid in station_train.keys():
            station_train[cid].append(sid)
        else:
            station_train[cid] = list()
            station_train[cid].append(sid)

    for k, v in station_train.items():

        if len(v) > 5:
            v = list(set(v))
            v = v[:5]
            station_train[k] = v

        if len(v) < 5:
            eng_name = list(cityRaw[cityRaw["city_id"] == k]["name_english"])[0]
            tmp = list(pd.read_csv("database/station/station_{}.csv".format(eng_name), dtype=object)["sid"])
            for removed in v:
                tmp.remove(removed)
            random.shuffle(tmp)
            station_train[k].append(tmp[0])

    station_train = [v for k, v in station_train.items()]

    '''
    featureData_t = (local_static, local_seq, others_static, others_seq)_t
    labelData_t = target_t
    '''

    dataNum = 0
    source_location = list()
    for i in range(len(station_train)):

        # location of local city
        sid_local = station_train[i][0]
        did_local = list(stationRaw[stationRaw["station_id"] == sid_local]["district_id"])[0]
        cid_local = list(districtRaw[districtRaw["district_id"] == did_local]["city_id"])[0]
        lat_local = float(cityRaw[cityRaw["city_id"] == cid_local]["latitude"])
        lon_local = float(cityRaw[cityRaw["city_id"] == cid_local]["longitude"])
        source_location.append((lat_local, lon_local))

        # location of other cities
        others_city = list()
        for j in range(len(station_train)):

            if i == j:
                continue

            sid = station_train[j][0]
            did = list(stationRaw[stationRaw["station_id"] == sid]["district_id"])[0]
            cid = list(districtRaw[districtRaw["district_id"] == did]["city_id"])[0]
            lat = float(cityRaw[cityRaw["city_id"] == cid]["latitude"])
            lon = float(cityRaw[cityRaw["city_id"] == cid]["longitude"])
            result = get_dist_angle(lat_local, lon_local, lat, lon)
            others_city.append([result["distance"], result["azimuth1"]])

        others_city = np.array(others_city)
        minimum = others_city.min(axis=0, keepdims=True)
        maximum = others_city.max(axis=0, keepdims=True)
        others_city = (others_city - minimum) / (maximum - minimum)
        others_city = list(map(lambda x: list(x), others_city))

        for station_local in station_train[i]:

            # output
            out_local_static = list()
            out_local_seq = list()
            out_others_static = list()
            out_others_seq = list()
            out_others_city = list()
            out_target = list()

            '''
            calculate distance and angle of other stations from local stations
            '''
            # lat, lon of local station
            lat_local = float(stationData[stationData["sid"] == station_local]["lat"])
            lon_local = float(stationData[stationData["sid"] == station_local]["lon"])

            # distance and angle
            geoVect = list()
            for j in range(len(station_train)):

                if i == j:
                    continue

                for station_others_j in station_train[j]:
                    lat = float(stationData[stationData["sid"] == station_others_j]["lat"])
                    lon = float(stationData[stationData["sid"] == station_others_j]["lon"])
                    result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
                    geoVect.append([result["distance"], result["azimuth1"]])

            # normalization others' location
            geoVect = np.array(geoVect)
            minimum = geoVect.min(axis=0, keepdims=True)
            maximum = geoVect.max(axis=0, keepdims=True)
            geoVect = (geoVect - minimum) / (maximum - minimum)
            geoVect = list(map(lambda x: list(x), geoVect))

            # add geoVect to static data
            others_static = list()
            idx = 0
            for j in range(len(station_train)):

                if i == j:
                    continue

                others_static_j = list()
                for station_others_j in station_train[j]:
                    others_static_j.append(staticData[station_others_j] + geoVect[idx])
                    idx += 1
                others_static.append(others_static_j)

            '''
            concut meteorological data with aqi data of seqData of others
            '''
            seqData_others = dict()
            for j in range(len(station_train)):

                if i == j:
                    continue

                for station_others_j in station_train[j]:
                    m = _pickle.loads(_pickle.dumps(meteorologyData[station_others_j], -1))  # _pickleを使った高速コピー
                    a = _pickle.loads(_pickle.dumps(aqiData[station_others_j], -1))  # _pickleを使った高速コピー
                    for k in range(len(m)):
                        for l in range(len(m[k])):
                            m[k][l] += a[k][l]
                    seqData_others[station_others_j] = m

            '''
            local data and target data
            '''
            local_static = staticData[station_local]
            local_seq = meteorologyData[station_local]
            target = targetData[station_local]
            dataNum = len(target)

            for t in range(dataNum):

                others_seq = list()

                for j in range(len(station_train)):

                    if i == j:
                        continue

                    others_seq_j = list()
                    for station_others_j in station_train[j]:
                        others_seq_j.append(seqData_others[station_others_j][t])
                    others_seq.append(others_seq_j)

                out_local_static.append(local_static)
                out_local_seq.append(local_seq[t])
                out_others_static.append(others_static)
                out_others_seq.append(others_seq)
                out_others_city.append(others_city)
                out_target.append(target[t])

            out_set = (out_local_static, out_local_seq, out_others_static, out_others_seq, out_others_city, out_target)

            cityCode = str(i).zfill(3)
            stationCode = str(station_train[i].index(station_local)).zfill(3)

            with bz2.BZ2File("{}/train_{}{}.pkl.bz2".format(savePath_train, cityCode, stationCode), 'wb', compresslevel=9) as fp:
                fp.write(pickle.dumps(out_set))
                print("* save train_{}{}.pkl.bz2".format(cityCode, stationCode))

            del out_local_seq, out_local_static, out_others_seq, out_others_static, out_others_city, out_target, out_set

    with open("{}/fileNum.pkl".format(savePath_train), "wb") as fp:
        pickle.dump({"station": len(station_train[0]), "city": len(station_train), "time": dataNum}, fp)

    '''
    test data
    '''

    dataNum = 0
    source_location = list()
    for i in range(len(station_train)):

        # location of local city
        sid_local = station_train[i][0]
        did_local = list(stationRaw[stationRaw["station_id"] == sid_local]["district_id"])[0]
        cid_local = list(districtRaw[districtRaw["district_id"] == did_local]["city_id"])[0]
        lat_local = float(cityRaw[cityRaw["city_id"] == cid_local]["latitude"])
        lon_local = float(cityRaw[cityRaw["city_id"] == cid_local]["longitude"])
        source_location.append((lat_local, lon_local))

    testNum = pickle.load(open("{}/fileNum.pkl".format(dataPath_test), "rb"))["test"]
    for i in range(testNum):

        # output
        out_others_static = list()
        out_others_seq = list()
        out_others_city = list()

        '''
        calculate distance and angle of other source cities from the target city
        '''

        tmp = pickle.load(bz2.BZ2File("{}/test_{}.pkl.bz2".format(dataPath_test, str(i).zfill(3)), 'rb'))
        out_local_static, out_local_seq, out_target, tmp = tmp[0], tmp[1], tmp[4], tmp[0][0]

        # location of local city
        sid_local = [k for k, v in staticData.items() if v == tmp][0]
        did_local = list(stationRaw[stationRaw["station_id"] == sid_local]["district_id"])[0]
        cid_local = list(districtRaw[districtRaw["district_id"] == did_local]["city_id"])[0]
        lat_local = float(cityRaw[cityRaw["city_id"] == cid_local]["latitude"])
        lon_local = float(cityRaw[cityRaw["city_id"] == cid_local]["longitude"])

        others_city = list()
        max_index = 0
        max_distance = 0
        for j in range(len(source_location)):
            lat, lon = source_location[j]
            result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
            others_city.append([result["distance"], result["azimuth1"]])
            if result["distance"] > max_distance:
                max_distance = result["distance"]
                max_index = j

        station_train_copy = station_train.copy()
        others_city.remove(others_city[max_index])
        station_train_copy.remove(station_train_copy[max_index])
        others_city = np.array(others_city)
        minimum = others_city.min(axis=0, keepdims=True)
        maximum = others_city.max(axis=0, keepdims=True)
        others_city = (others_city - minimum) / (maximum - minimum)
        others_city = list(map(lambda x: list(x), others_city))

        '''
        calculate distance and angle of other stations from local stations
        '''
        # lat, lon of local station
        lat_local = float(stationData[stationData["sid"] == sid_local]["lat"])
        lon_local = float(stationData[stationData["sid"] == sid_local]["lon"])

        # distance and angle
        geoVect = list()
        for j in range(len(station_train_copy)):
            for station_others_j in station_train_copy[j]:
                lat = float(stationData[stationData["sid"] == station_others_j]["lat"])
                lon = float(stationData[stationData["sid"] == station_others_j]["lon"])
                result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
                geoVect.append([result["distance"], result["azimuth1"]])

        # normalization others' location
        geoVect = np.array(geoVect)
        minimum = geoVect.min(axis=0, keepdims=True)
        maximum = geoVect.max(axis=0, keepdims=True)
        geoVect = (geoVect - minimum) / (maximum - minimum)
        geoVect = list(map(lambda x: list(x), geoVect))

        # add geoVect to static data
        others_static = list()
        idx = 0
        for j in range(len(station_train_copy)):
            others_static_j = list()
            for station_others_j in station_train_copy[j]:
                others_static_j.append(staticData[station_others_j] + geoVect[idx])
                idx += 1
            others_static.append(others_static_j)

        '''
        concut meteorological data with aqi data of seqData of others
        '''
        seqData_others = dict()
        for j in range(len(station_train_copy)):
            for station_others_j in station_train_copy[j]:
                m = _pickle.loads(_pickle.dumps(meteorologyData[station_others_j], -1))  # _pickleを使った高速コピー
                a = _pickle.loads(_pickle.dumps(aqiData[station_others_j], -1))  # _pickleを使った高速コピー
                for k in range(len(m)):
                    for l in range(len(m[k])):
                        m[k][l] += a[k][l]
                seqData_others[station_others_j] = m

        '''
        output set
        '''
        dataNum = len(out_target)
        for t in range(dataNum):

            others_seq = list()
            for j in range(len(station_train_copy)):
                others_seq_j = list()
                for station_others_j in station_train_copy[j]:
                    others_seq_j.append(seqData_others[station_others_j][t])
                others_seq.append(others_seq_j)

            out_others_static.append(others_static)
            out_others_seq.append(others_seq)
            out_others_city.append(others_city)

        out_set = (out_local_static, out_local_seq, out_others_static, out_others_seq, out_others_city, out_target)

        with bz2.BZ2File("{}/test_{}.pkl.bz2".format(savePath_test, str(i).zfill(3)), 'wb', compresslevel=9) as fp:
            fp.write(pickle.dumps(out_set))
            print("* save test_{}.pkl.bz2".format(str(i).zfill(3)))

        del out_local_seq, out_local_static, out_others_seq, out_others_static, out_others_city, out_target, out_set

    with open("{}/fileNum.pkl".format(savePath_test), "wb") as fp:
       pickle.dump({"station": len(station_train[0]), "city": len(station_train)+1, "time": dataNum}, fp)

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
    batch_size = 32
    epochs = 100
    lr = 0.01
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
    patience = epochs
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # log
    logs = list()

    # dataset path
    dataPath = pickle.load(open("tmp/trainPath.pkl", "rb"))

    # the number which the train/validation dataset was divivded into
    trainNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["train"]
    #validNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["valid"]

    # start training
    for step in range(int(epochs)):

        # train
        for idx in range(trainNum):

            epoch_loss = list()

            selector = "/train_{}.pkl.bz2".format(str(idx).zfill(3))
            trainData = MyDataset_ADAIN(pickle.load(bz2.BZ2File(dataPath + selector, 'rb')))
            for batch_i in torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True):

                print("\t|- train batch loss: ", end="")

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

        # # validate
        # print("\t\t|- validation : ", end="")
        # rmse = list()
        # accuracy = list()
        # model.eval()
        # for idx in range(validNum):
        #     selector = "/valid_{}.pkl.bz2".format(str(idx).zfill(3))
        #     validData = MyDataset_ADAIN(pickle.load(bz2.BZ2File(dataPath + selector, 'rb')))
        #     rmse_i, accuracy_i = validate_ADAIN(model, validData)
        #     rmse.append(rmse_i)
        #     accuracy.append(accuracy_i)
        # # model.train()
        #
        # # calculate validation loss
        # rmse = np.average(rmse)
        # accuracy = np.average(accuracy)
        # log = {'epoch': step, 'validation rmse': rmse, 'validation accuracy': accuracy}
        # logs.append(log)
        # print("rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

            # evaluate
            model.eval()
            rmse, accuracy = midium_evaluate_ADAIN(model)
            model.train()
            log = {'epoch': step, 'train_rmse': epoch_loss, 'test_rmse': rmse}
            logs.append(log)
            print("\t\t|- rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

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

def midium_evaluate_ADAIN(model):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for evaluation
    result = list()
    result_label = list()

    # dataset path
    dataPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # the number which the test dataset was divided into
    testNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["test"]

    batch_size = 200

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

def _objective_HARADA(trial):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyper parameters for tuning
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    # epochs = trial.suggest_discrete_uniform("epochs", 1, 5, 1)
    # lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    # wd = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    # hyper parameters for constance
    alpha = 0.5
    beta = 1.0 - alpha
    gamma = 1.0
    eta = 1.0
    batch_size = 32
    epochs = 200
    lr = 0.01
    wd = 0.0

    # dataset path
    trainPath = pickle.load(open("tmp/trainPath.pkl", "rb"))
    testPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # input dimension
    inputDim = pickle.load(open("datatmp/inputDim.pkl", "rb"))
    cityNum = pickle.load(open("{}/fileNum.pkl".format(trainPath), "rb"))["city"]
    stationNum = pickle.load(open("{}/fileNum.pkl".format(trainPath), "rb"))["station"]

    # model
    model = _HARADA(inputDim_local_static=inputDim["local_static"],
                   inputDim_local_seq=inputDim["local_seq"],
                   inputDim_others_static=inputDim["others_static"],
                   inputDim_others_seq=inputDim["others_seq"],
                   cityNum=cityNum,
                   stationNum=stationNum)

    # GPU or CPU
    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # loss function
    criterion_mse = nn.MSELoss()
    criterion_mmd = SamplesLoss("gaussian", blur=0.5)

    # initialize the early stopping object
    patience = epochs
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # log
    logs = list()

    # mmd data
    mmdData = list()
    testData = list()
    testData_idx = list()
    for i in range(1):
        tmp = pickle.load(bz2.BZ2File("{}/test_{}.pkl.bz2".format(testPath, str(i).zfill(3)), 'rb'))
        mmdData.append(MyDataset_MMD(tmp[:2]))
        testData.append(MyDataset_HARADA(tmp))
        testData_idx.append(pickle.load(open("{}/test_{}_idx.pkl".format(testPath, str(i).zfill(3)), 'rb')))
    print("mmd data was loaded")

    # start training
    for step in range(int(epochs)):

        epoch_loss = list()
        stationSelector = [random.randrange(0, 5) for i in range(cityNum)]

        #for idx in range(len(stationSelector)):
        for idx in range(1):

            repeat_loss = list()
            #selectPath = "{}/train_{}{}.pkl.bz2".format(trainPath, str(idx).zfill(3), str(stationSelector[idx]).zfill(3))
            selectPath = "{}/train_{}{}.pkl.bz2".format(trainPath, str(idx).zfill(3), str(0).zfill(3))
            trainData = MyDataset_HARADA(pickle.load(bz2.BZ2File(selectPath, "rb")))
            trainData = list(torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=False))

            #selectPath = "{}/train_{}{}_idx.pkl".format(trainPath, str(idx).zfill(3), str(stationSelector[idx]).zfill(3))
            selectPath = "{}/train_{}{}_idx.pkl".format(trainPath, str(idx).zfill(3), str(0).zfill(3))
            local_index = pickle.load(open(selectPath, "rb"))

            for batch_i in range(len(trainData)):

                print("\t|- mid-loss: ", end="")

                optimizer.zero_grad()

                # batch data
                batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city, batch_target = trainData[batch_i]

                # to GPU
                batch_local_static = batch_local_static.to(device)
                batch_local_seq = batch_local_seq.to(device)
                batch_others_static = batch_others_static.to(device)
                batch_others_seq = batch_others_seq.to(device)
                batch_others_city = batch_others_city.to(device)
                batch_target = batch_target.to(device)

                # predict
                y_moe, y_mtl, y_mmd, etp = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city, local_index)

                # loss append
                batch_loss_moe = criterion_mse(y_moe, batch_target)
                tmp = np.sqrt(float(batch_loss_moe.item()))
                repeat_loss.append(tmp)

                batch_loss_mtl = 0
                for y_mtl_i in y_mtl:
                    batch_loss_mtl += (1/(len(y_mtl))) * criterion_mse(y_mtl_i, batch_target)

                # mmd target
                batch_local_static = list()
                batch_local_seq = list()
                for i in range(stationNum):
                    mmdData_i = list(torch.utils.data.DataLoader(mmdData[i], batch_size=batch_size, shuffle=False))
                    mmdData_i = mmdData_i[batch_i]
                    batch_local_static.append(mmdData_i[0])
                    batch_local_seq.append(mmdData_i[1])

                # stack
                batch_local_static = torch.cat(batch_local_static, dim=0)
                batch_local_seq = torch.cat(batch_local_seq, dim=0)

                # to GPU
                batch_local_static = batch_local_static.to(device)
                batch_local_seq = batch_local_seq.to(device)

                # calculate mmd target
                mmd_target = model.encode(batch_local_static, batch_local_seq)

                # mmd source
                batch_loss_mmd = criterion_mmd(mmd_target, y_mmd) ** 2

                # loss (multi-task learning)
                batch_loss = (alpha * batch_loss_moe) + (beta * batch_loss_mtl) + (gamma * batch_loss_mmd) + (eta * etp)
                batch_loss.backward()
                optimizer.step()

                print("{}, total: {}".format(str(tmp), str(float(batch_loss.item()))))

            repeat_loss = np.mean(repeat_loss)
            epoch_loss.append(repeat_loss)
            print("\t\t|- epoch loss: {}".format(str(repeat_loss)))

            # evaluate
            model.eval()
            rmse, accuracy = _midium_evaluate_HARADA(model, testData, testData_idx)
            model.train()
            log = {'epoch': step, 'train_rmse': np.mean(epoch_loss), 'test_rmse': rmse}
            logs.append(log)
            print("\t\t|- rmse: {}, accuracy: {}".format(str(rmse), str(accuracy)))

            # early stopping
            early_stopping(rmse, model)
            if early_stopping.early_stop:
                print("\t\t\tEarly stopping")
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


def _midium_evaluate_HARADA(model, testData, testData_idx):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 128
    iteration = 0

    testPath = pickle.load(open("tmp/testPath.pkl", "rb"))
    stationNum = pickle.load(open("{}/fileNum.pkl".format(testPath), "rb"))["station"]

    # for evaluation
    result = list()
    result_label = list()

    for i in range(stationNum):
        local_idx = testData_idx[i]
        for batch_i in torch.utils.data.DataLoader(testData[i], batch_size=batch_size, shuffle=False):

            # batch data
            batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city, batch_target = batch_i

            # to GPU
            batch_local_static = batch_local_static.to(device)
            batch_local_seq = batch_local_seq.to(device)
            batch_others_static = batch_others_static.to(device)
            batch_others_seq = batch_others_seq.to(device)
            batch_others_city = batch_others_city.to(device)

            # predict
            with torch.no_grad():
                y_moe, y_mtl, y_mmd, etp = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city, local_idx)
                pred = y_moe.to("cpu")

                # evaluate
                pred = list(map(lambda x: x[0], pred.data.numpy()))
                batch_target = list(map(lambda x: x[0], batch_target.data.numpy()))
                result += pred
                result_label += batch_target

                iteration += len(batch_target)

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)

    return rmse, accuracy

def objective_HARADA(trial):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyper parameters for tuning
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    # epochs = trial.suggest_discrete_uniform("epochs", 1, 5, 1)
    # lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    # wd = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    # hyper parameters for constance
    alpha = 0.5
    beta = 1.0 - alpha
    gamma = 1.0
    eta = 1.0
    batch_size = 32
    epochs = 200
    lr = 0.01
    wd = 0.0

    # dataset path
    trainPath = pickle.load(open("tmp/trainPath.pkl", "rb"))
    testPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # input dimension
    inputDim = pickle.load(open("datatmp/inputDim.pkl", "rb"))
    cityNum = pickle.load(open("{}/fileNum.pkl".format(trainPath), "rb"))["city"]
    stationNum = pickle.load(open("{}/fileNum.pkl".format(trainPath), "rb"))["station"]

    # model
    model = HARADA(inputDim_local_static=inputDim["local_static"],
                   inputDim_local_seq=inputDim["local_seq"],
                   inputDim_others_static=inputDim["others_static"],
                   inputDim_others_seq=inputDim["others_seq"],
                   cityNum=cityNum-1,
                   stationNum=stationNum)

    # GPU or CPU
    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # loss function
    criterion_mse = nn.MSELoss()
    criterion_mmd = SamplesLoss("gaussian", blur=0.5)

    # initialize the early stopping object
    patience = epochs
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # log
    logs = list()

    # mmd data
    mmdData = list()
    testData = list()
    for i in range(stationNum):
        tmp = pickle.load(bz2.BZ2File("{}/test_{}.pkl.bz2".format(testPath, str(i).zfill(3)), 'rb'))
        mmdData.append(MyDataset_MMD(tmp[:2]))
        testData.append(MyDataset_HARADA(tmp))
    print("mmd data was loaded")

    # start training
    for step in range(int(epochs)):

        epoch_loss = list()
        stationSelector = [random.randrange(0, 5) for i in range(cityNum)]

        for idx in range(len(stationSelector)):

            repeat_loss = list()
            selectPath = "{}/train_{}{}.pkl.bz2".format(trainPath, str(idx).zfill(3), str(stationSelector[idx]).zfill(3))
            trainData = MyDataset_HARADA(pickle.load(bz2.BZ2File(selectPath, "rb")))
            trainData = list(torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=False))

            for batch_i in range(len(trainData)):

                print("\t|- mid-loss: ", end="")

                optimizer.zero_grad()

                # batch data
                batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city, batch_target = trainData[batch_i]

                # to GPU
                batch_local_static = batch_local_static.to(device)
                batch_local_seq = batch_local_seq.to(device)
                batch_others_static = batch_others_static.to(device)
                batch_others_seq = batch_others_seq.to(device)
                batch_others_city = batch_others_city.to(device)
                batch_target = batch_target.to(device)

                # predict
                y_moe, y_mtl, y_mmd, etp = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city)

                # loss append
                batch_loss_moe = criterion_mse(y_moe, batch_target)
                tmp = np.sqrt(float(batch_loss_moe.item()))
                repeat_loss.append(tmp)

                batch_loss_mtl = 0
                for y_mtl_i in y_mtl:
                    batch_loss_mtl += (1/(len(y_mtl))) * criterion_mse(y_mtl_i, batch_target)

                # mmd target
                batch_local_static = list()
                batch_local_seq = list()
                for i in range(stationNum):
                    mmdData_i = list(torch.utils.data.DataLoader(mmdData[i], batch_size=batch_size, shuffle=False))
                    mmdData_i = mmdData_i[batch_i]
                    batch_local_static.append(mmdData_i[0])
                    batch_local_seq.append(mmdData_i[1])

                # stack
                batch_local_static = torch.cat(batch_local_static, dim=0)
                batch_local_seq = torch.cat(batch_local_seq, dim=0)

                # to GPU
                batch_local_static = batch_local_static.to(device)
                batch_local_seq = batch_local_seq.to(device)

                # calculate mmd target
                mmd_target = model.encode(batch_local_static, batch_local_seq)

                # mmd source
                batch_loss_mmd = criterion_mmd(mmd_target, y_mmd) ** 2

                # loss (multi-task learning)
                batch_loss = (alpha * batch_loss_moe) + (beta * batch_loss_mtl) + (gamma * batch_loss_mmd) + (eta * etp)
                batch_loss.backward()
                optimizer.step()

                print("{}, total: {}".format(str(tmp), str(float(batch_loss.item()))))

            repeat_loss = np.mean(repeat_loss)
            epoch_loss.append(repeat_loss)
            print("\t\t|- epoch loss: {}".format(str(repeat_loss)))

            # evaluate
            model.eval()
            rmse, accuracy = midium_evaluate_HARADA(model, testData)
            model.train()
            log = {'epoch': step, 'train_rmse': np.mean(epoch_loss), 'test_rmse': rmse}
            logs.append(log)
            print("\t\t|- rmse: {}, accuracy: {}".format(str(rmse), str(accuracy)))

            # early stopping
            early_stopping(rmse, model)
            if early_stopping.early_stop:
                print("\t\t\tEarly stopping")
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

def midium_evaluate_HARADA(model, testData):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 128
    iteration = 0

    testPath = pickle.load(open("tmp/testPath.pkl", "rb"))
    stationNum = pickle.load(open("{}/fileNum.pkl".format(testPath), "rb"))["station"]

    # for evaluation
    result = list()
    result_label = list()

    for i in range(stationNum):
        for batch_i in torch.utils.data.DataLoader(testData[i], batch_size=batch_size, shuffle=False):

            # batch data
            batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city, batch_target = batch_i

            # to GPU
            batch_local_static = batch_local_static.to(device)
            batch_local_seq = batch_local_seq.to(device)
            batch_others_static = batch_others_static.to(device)
            batch_others_seq = batch_others_seq.to(device)
            batch_others_city = batch_others_city.to(device)

            # predict
            with torch.no_grad():
                y_moe, y_mtl, y_mmd, etp = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city)
                pred = y_moe.to("cpu")

                # evaluate
                pred = list(map(lambda x: x[0], pred.data.numpy()))
                batch_target = list(map(lambda x: x[0], batch_target.data.numpy()))
                result += pred
                result_label += batch_target

                iteration += len(batch_target)

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)

    return rmse, accuracy

def evaluate_HARADA(model_state_dict):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 32
    iteration = 0

    # dataset path
    dataPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # input dimension
    inputDim = pickle.load(open("datatmp/inputDim.pkl", "rb"))
    cityNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["city"]
    stationNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["station"]
    dataNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["time"]

    # model
    model = HARADA(inputDim_local_static=inputDim["local_static"],
                   inputDim_local_seq=inputDim["local_seq"],
                   inputDim_others_static=inputDim["others_static"],
                   inputDim_others_seq=inputDim["others_seq"],
                   cityNum=cityNum-1,
                   stationNum=stationNum)

    model.load_state_dict(model_state_dict)
    model = model.to(device)

    # evaluate mode
    model.eval()

    # for evaluation
    result = list()
    result_label = list()

    for station_id in range(stationNum):
        selectPath = "/test_{}.pkl.bz2".format(str(station_id).zfill(3))
        testData = MyDataset_HARADA(pickle.load(bz2.BZ2File(dataPath + selectPath, 'rb')))
        for batch_i in torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False):

            # batch data
            batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city, batch_target = batch_i

            # to GPU
            batch_local_static = batch_local_static.to(device)
            batch_local_seq = batch_local_seq.to(device)
            batch_others_static = batch_others_static.to(device)
            batch_others_seq = batch_others_seq.to(device)
            batch_others_city = batch_others_city.to(device)

            # predict
            pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city)[0]
            pred = pred.to("cpu")

            # evaluate
            pred = list(map(lambda x: x[0], pred.data.numpy()))
            batch_target = list(map(lambda x: x[0], batch_target.data.numpy()))
            result += pred
            result_label += batch_target

            iteration += len(batch_target)
            print("\t|- iteration %d / %d" % (iteration, dataNum * stationNum))

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
    batch_size = 32
    epochs = 30
    lr = 0.01
    wd = 0.005

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
    patience = epochs
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
        result += aqiData_source[1000:3000]
        result_label += aqiData_target[1000:3000]

    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)

    return rmse, accuracy

def repeat_KNN(TARGET, start, end):

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
    result = dict()
    result_label = dict()
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

        # aqi data of target city
        aqiData_target = aqiData[target_station]

        # get K nearest neighbors
        distance = sorted(distance.items(), key=lambda x: x[1])

        for K in range(start, end):
            nearest = list(map(lambda x: x[0], distance[:K]))

            # agi data of source cities
            aqiData_source = list()
            for source_station_i in nearest:
                aqiData_source.append(aqiData[source_station_i])
            aqiData_source = list(np.mean(np.array(aqiData_source), axis=0))

            # evaluate
            if str(K) in result.keys():
                result += aqiData_source
                result_label += aqiData_target
            else:
                result[str(K)] = list()
                result_label[str(K)] = list()
                result[str(K)] += aqiData_source
                result_label[str(K)] += aqiData_target

    for k in result.keys():
        rmse = np.sqrt(mean_squared_error(result[k], result_label[k]))
        with open("result/{}Test19KNN_analysis.csv".format(TARGET), "a") as fp:
            fp.write("{},{}\n".format(str(k), str(rmse)))

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
        print(distance[:K])

        # nearest = list(map(lambda x: x[0], distance[:K]))
        #
        # # agi data of source cities
        # aqiData_source = list()
        # for source_station_i in nearest:
        #     aqiData_source.append(aqiData[source_station_i])
        #
        # # aqi data of target city
        # aqiData_target = aqiData[target_station]
        #
        # # evaluate
        # aqiData_source_mean = list(np.mean(np.array(aqiData_source), axis=0))
        # rmse = np.sqrt(mean_squared_error(aqiData_target, aqiData_source_mean))
        #
        # with open("result/K{}_idx{}.csv".format(str(K), str(idx).zfill(3)), "w") as outfile:
        #     outfile.write("RMSE={}\n".format(str(rmse)))
        #     outfile.write("{},mean,{}\n".format(",".join(nearest), target_station))
        #     for t in range(len(aqiData_target)):
        #         tmp = list()
        #         for s in range(len(nearest)):
        #             tmp.append(str(aqiData_source[s][t]))
        #         tmp.append(str(aqiData_source_mean[t]))
        #         tmp.append(str(aqiData_target[t]))
        #         outfile.write("{}\n".format(",".join(tmp)))

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
        aqiData_source = list(np.sum(np.array(aqiData_source), axis=0))
        result += aqiData_source[1000:3000]
        result_label += aqiData_target[1000:3000]

    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)

    return rmse, accuracy