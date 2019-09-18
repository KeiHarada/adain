# to run on server
import sys
sys.path.append("/home/harada/Documents/WorkSpace/adain")
sys.path.append("/home")

import re
import requests
import json
import xml.etree.ElementTree as ET
import shapely.geometry
import pyproj
import pandas as pd
import argparse
import overpy
from time import sleep
from source.utility import citycode
from source.utility import Color
from source.utility import get_road_over_the_city
from source.utility import get_grid_id

#cities = ["beijing", "shenzhen", "tianjin", "guangzhou"]
cities = list(pd.read_csv("rawdata/zheng2015/city.csv")["name_english"])
print(cities)

def grid(scale, gridsize):

    for city in cities:
        with open("rawdata/osm/"+scale+"/osm_"+scale+"_"+city, "r") as infile:

            # Create city information file
            root = ET.ElementTree(file=infile)
            ll = dict(root.getroot()[2].attrib)
            with open("database/city/city_"+city+".csv", "w") as outfile:
                outfile.write("minlat,minlon,maxlat,maxlon\n")
                outfile.write(ll["minlat"]+","+ll["minlon"]+","+ll["maxlat"]+","+ll["maxlon"]+"\n")

            # Set up projections
            p_ll = pyproj.Proj(init='epsg:4326')
            p_mt = pyproj.Proj(init='epsg:3857')  # metric; same as EPSG:900913

            # Create corners of rectangle to be transformed to a grid
            WS = shapely.geometry.Point((float(ll["minlon"]), float(ll["minlat"]))) #(x,y)=(lon,lat)
            EN = shapely.geometry.Point((float(ll["maxlon"]), float(ll["maxlat"]))) #(x,y)=(lon,lat)

            # Project corners to target projection
            WS = pyproj.transform(p_ll, p_mt, WS.x, WS.y)  # WS = (min_x,min_y)
            EN = pyproj.transform(p_ll, p_mt, EN.x, EN.y)  # EN = (max_x,max_y)

            # Iterate over 2D area
            grid_id = 0
            with open("database/grid/grid_" + city + ".csv", "w") as outfile:
                outfile.write("gid,minlat,minlon,maxlat,maxlon\n")
                x = WS[0]
                while x <= EN[0]:
                    y = WS[1]
                    while y <= EN[1]:
                        ws = pyproj.transform(p_mt, p_ll, x, y)
                        en = pyproj.transform(p_mt, p_ll, x+gridsize, y+gridsize)
                        outfile.write(str(grid_id)+","+"{},{},{},{}\n".format(ws[1], ws[0], en[1], en[0]))
                        y += gridsize
                        grid_id += 1
                    x += gridsize

def station(scale):

    for city in cities:

        # input
        with open("rawdata/zheng2015/station.csv", "r") as infile:

            if scale == "develop":
                scale = 5
            if scale == "small":
                scale = 10
            if scale == "large":
                scale = 10000

            # output
            with open("database/station/station_" + city + ".csv", "w") as outfile:
                outfile.write("sid,lat,lon,did,gid\n")
                pattern = citycode(name=city, scale="station")
                station_counter = 0
                for line in infile.readlines()[1:]:
                    line = line.strip().split(",")

                    if re.match(pattern, line[0]):
                        gid = "NULL"
                        line = [line[0], line[3], line[4], line[5], gid]
                        outfile.write(",".join(line) + "\n")
                        station_counter += 1

                    if station_counter >= scale:
                        break

def station_grid():

    for city in cities:

        # input
        with open("rawdata/zheng2015/station.csv", "r") as infile:

            # ll
            with open("database/city/city_"+city+".csv", "r") as cityfile:
                minlat, minlon, maxlat, maxlon = [float(s) for s in cityfile.readlines()[1].strip().split(",")]

                # output
                with open("database/station/station_" + city + ".csv", "w") as outfile:
                    outfile.write("sid,lat,lon,did,gid\n")
                    pattern = citycode(name=city, scale="station")
                    for line in infile.readlines()[1:]:
                        line = line.strip().split(",")
                        if re.match(pattern, line[0]):
                            if minlat < float(line[3]) < maxlat and minlon < float(line[4]) < maxlon:
                                gid = get_grid_id(city=city, lat=float(line[3]), lon=float(line[4]))
                                line = [line[0], line[3], line[4], line[5], gid]
                                outfile.write(",".join(line)+"\n")

def aqi():

    for city in cities:

        # station list
        stationlist = list()
        with open("database/station/station_"+city+".csv", "r") as stationfile:
            for station in stationfile.readlines()[1:]:
                stationlist.append(station.strip().split(",")[0])

        # input
        with open("rawdata/zheng2015/airquality.csv", "r") as infile:

            # output
            with open("database/aqi/aqi_" + city + ".csv", "w") as outfile:
                outfile.write("sid,time,pm25,pm10,no2,co,o3,so2\n")
                pattern = citycode(name=city, scale="station")
                for line in infile.readlines()[1:]:
                    line = line.strip().split(",")
                    if re.match(pattern, line[0]):
                        if line[0] in stationlist:
                            outfile.write(",".join(line)+"\n")

def meteorology():

    for city in cities:

        # district list
        districtlist = list()
        with open("database/station/station_"+city+".csv", "r") as stationfile:
            for station in stationfile.readlines()[1:]:
                districtlist.append(station.strip().split(",")[3])

        # input
        with open("rawdata/zheng2015/meteorology.csv", "r") as infile:

            # output
            with open("database/meteorology/meteorology_" + city + ".csv", "w") as outfile:
                outfile.write("did,time,weather,temperature,pressure,humidity,wind_speed,wind_direction\n")
                pattern = citycode(name=city, scale="district")
                for line in infile.readlines()[1:]:
                    line = line.strip().split(",")
                    if re.match(pattern, line[0]):
                        if line[0] in districtlist:
                            outfile.write(",".join(line)+"\n")

def unknown():

    for city in cities:

        # input
        with open("rawdata/zheng2015/meteorology.csv", "r") as infile:

            # output
            with open("database/meteorology/unknown/meteorology_" + city + ".csv", "w") as outfile:
                outfile.write("cid,time,weather,temperature,pressure,humidity,wind_speed,wind_direction\n")
                pattern = citycode(name=city, scale="city")
                for line in infile.readlines()[1:]:
                    line = line.strip().split(",")
                    if re.match(pattern, line[0]):
                        outfile.write(",".join(line)+"\n")

def createCategory_tree(subtree, category_dict, category):

    if len(subtree["categories"]) == 0:
        category_dict[subtree["name"]] = category
        return category_dict

    for item in subtree["categories"]:

        category_dict = createCategory_tree(item, category_dict, category)
        category_dict[item["name"]] = category

    return category_dict

def poi(key, secret, radius):

    # create category list
    category_dict = dict()
    poi_counter = dict()
    url = "https://api.foursquare.com/v2/venues/categories"
    params = dict(
        client_id=args.key,
        client_secret=args.secret,
        v="20140401"
    )
    response = requests.get(url=url, params=params)
    response = json.loads(response.text)["response"]
    with open("database/poi/category.csv", "w") as category_file:
        category_file.write("https://developer.foursquare.com/docs/resources/categories\n")
        for item in response["categories"]:
            category_file.write(item["name"]+"\n")
            poi_counter[item["name"]] = 0
            category_dict = createCategory_tree(item, category_dict, item["name"])

    # get poi data
    for city in cities:

        no_poi_stations = list()

        with open("database/poi/poi_"+city+".csv", "w") as outfile:
            outfile.write("sid,{}\n".format(",".join(list(poi_counter.keys()))))

            for line in open("database/station/station_"+city+".csv", "r").readlines()[1:]:

                _poi_list = poi_counter.copy()

                sid, lat, lon, did, gid = line.strip().split(",")

                url = "https://api.foursquare.com/v2/venues/search"
                params = dict(
                    client_id=key,
                    client_secret=secret,
                    intent="browse",
                    ll="{},{}".format(lat, lon),
                    radius=radius, #[m]
                    limit="10000",
                    v="20140401"
                )
                sleep(0.1)
                response = requests.get(url=url, params=params)
                response = json.loads(response.text)

                # check response
                if response["meta"]["code"] == 200:
                    response = response["response"]
                else:
                    print("error {}".format(str(response["metta"]["code"])))
                    exit()

                # count poi
                if len(response["venues"]) == 0:
                    no_poi_stations.append(sid)
                else:
                    for venue in response["venues"]:

                        # select primary category
                        for category in venue["categories"]:
                            if category["primary"] is True:
                                category = category["name"]

                        # select parent category
                        if category in category_dict:
                            _poi_list[category_dict[category]] += 1

                    # write into a file
                    _poi_list = map(str, list(_poi_list.values()))
                    outfile.write("{},{}\n".format(str(sid), ",".join(_poi_list)))

        # drop no poi stations from station file
        if len(no_poi_stations) > 0:
            stations = pd.read_csv("database/station/station_"+city+".csv", dtype=object)
            for _sid in no_poi_stations:
                stations = stations[stations["sid"] != _sid]
            stations.to_csv("database/station/station_"+city+".csv", index=False)

def road(radius):
    for city in cities:
        with open("database/road/road_" + city + ".csv", "w") as outfile:
            outfile.write("sid,motorway,trunk,others\n")

            for line in open("database/station/station_"+city+".csv", "r").readlines()[1:]:

                for i in range(1, 11):

                    try:
                        api = overpy.Overpass()
                        sid, lat, lon, did, gid = line.strip().split(",")
                        result = api.query("way(around:"+str(float(radius))+","+lat+","+lon+");out;")
                        road = dict(
                            motorway=0,
                            trunk=0,
                            others=0,
                            na=0
                        )
                        for way in result.ways:
                            highway = way.tags.get("highway", "n/a")
                            if highway == "n/a":
                                road["na"] += 1
                            elif highway in road:
                                road[highway] += 1
                            else:
                                road["others"] += 1
                        outfile.write(sid+","+str(road["motorway"])+","+str(road["trunk"])+","+str(road["others"])+"\n")
                    except overpy.exception as e:
                        print("error:{e} retry:{i}/10".format(e=e, i=i))
                        sleep(i * 5)
                        fg = False
                    else:
                        fg = True

                    if fg:
                        break

# def getPOI(url, params):
#
#     sleep(1.0)
#     response = requests.get(url=url, params=params)
#     response = json.loads(response.text)
#     print(response)
#     pois = set()
#
#     if len(response["response"]["venues"]) >= 50:
#         minlat, minlon = params["sw"].split(",")
#         maxlat, maxlon = params["ne"].split(",")
#         lat = (float(maxlat) - float(minlat))/2
#         lon = (float(maxlon) - float(minlon))/2
#
#         # top right
#         _params = params.copy()
#         _params["ne"] = "{},{}".format(maxlat, maxlon)
#         _params["sw"] = "{},{}".format(str(float(minlat)+lat), str(float(minlon)+lon))
#         poi = getPOI(url, _params)
#         pois |= poi
#
#         # top left
#         _params = params.copy()
#         _params["ne"] = "{},{}".format(maxlat, str(float(maxlon)-lon))
#         _params["sw"] = "{},{}".format(str(float(minlat)+lat), minlon)
#         poi = getPOI(url, _params)
#         pois |= poi
#
#         # bottom right
#         _params = params.copy()
#         _params["ne"] = "{},{}".format(str(float(maxlat)-lat), maxlon)
#         _params["sw"] = "{},{}".format(minlat, str(float(minlon)+lon))
#         poi = getPOI(url, _params)
#         pois |= poi
#
#         # bottom left
#         _params = params.copy()
#         _params["ne"] = "{},{}".format(str(float(maxlat)-lat), str(float(maxlon)-lon))
#         _params["sw"] = "{},{}".format(minlat, minlon)
#         poi = getPOI(url, _params)
#         pois |= poi
#
#     else:
#
#         for venue_id in list(map(lambda x: x["id"], response)):
#             url_venue = "https://api.foursquare.com/v2/venues/"+venue_id
#             params_venue = dict(
#                 client_id=params["key"],
#                 client_secret=params["secret"],
#                 v="20190101"
#             )
#
#             sleep(1.0)
#             response_venue = requests.get(url=url_venue, params=params_venue)
#             response_venue = json.loads(response_venue.text)
#             for item in response_venue:
#                 pois.add((item["id"], str(item["location"]["lat"]), str(item["location"]["lng"])))
#                 print((item["id"], str(item["location"]["lat"]), str(item["location"]["lng"])))
#
#     return pois

## grid base ##
# def road():
#
#     for city in cities:
#
#         #print(get_road_over_the_city(city))
#         with open("database/road/road_"+city+".csv", "w") as outfile:
#
#             outfile.write("gid,motorway,trunk,others\n")
#             for line in open("database/grid/grid_"+city+".csv", "r").readlines()[1:]:
#                 gid, minlat, minlon, maxlat, maxlon = line.strip().split(",")
#                 api = overpy.Overpass()
#                 result = api.query("way("+minlat+"," + minlon+"," + maxlat+"," + maxlon+");out;") #(minlat, minlon, maxlat, maxlon)
#                 road = dict(
#                     motorway=0,
#                     trunk=0,
#                     others=0,
#                     na=0
#                 )
#                 for way in result.ways:
#                     highway = way.tags.get("highway", "n/a")
#                     if highway == "n/a":
#                         road["na"] += 1
#                     elif highway in road:
#                         road[highway] += 1
#                     else:
#                         road["others"] += 1
#                 outfile.write(gid+","+str(road["motorway"])+","+str(road["trunk"])+","+str(road["others"])+"\n")

if __name__ == "__main__":

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("cityscale", help="please specify large or small, or develop", type=str)
    parser.add_argument("gridsize", help="please set me with meter scale", type=int)
    parser.add_argument("radius", help="please set me with meter scale", type=int)
    parser.add_argument("key", help="your key to access poi api", type=str)
    parser.add_argument("secret", help="your secret to access poi api", type=str)
    args = parser.parse_args()

    # print("map scale: "+args.cityscale)
    # print("\t|- grid data is build ... ", end="")
    # grid(args.cityscale, args.gridsize) # for demonstration
    # print(Color.GREEN + "OK" + Color.END)

    print("\t|- station data is build ... ", end="")
    station(args.cityscale)
    # station_grid() # for demonstration
    print(Color.GREEN + "OK" + Color.END)

    print("\t|- poi data is build ... ", end="")
    poi(args.key, args.secret, args.radius)
    print(Color.GREEN + "OK" + Color.END)

    print("\t|- road network data is build ... ", end="")
    road(args.radius)
    print(Color.GREEN + "OK" + Color.END)

    print("\t|- aqi data is build ... ", end="")
    aqi()
    print(Color.GREEN + "OK" + Color.END)

    print("\t|- meteorological data is build ... ", end="")
    meteorology()
    #unknown() # for demonstration
    print(Color.GREEN + "OK" + Color.END)