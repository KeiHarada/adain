import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import pandas as pd
import numpy as np
import datetime as dt
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from cartopy.io.img_tiles import OSM

from pprint import pprint

def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)

def test(ll):

    s = ll["minlat"]
    w = ll["minlon"]
    n = ll["maxlat"]
    e = ll["maxlon"]

    # 50m解像度用の陸地データ作成
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face',
                                            facecolor=cfeature.COLORS['land'])
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=140.))
    ax.set_extent([w, e, s, n])
    ax.add_feature(land_50m)
    ax.coastlines(resolution='50m', lw=0.5)
    ax.gridlines(xlocs=mticker.MultipleLocator(10),
                 ylocs=mticker.MultipleLocator(10),
                 linestyle='-',
                 color='gray')
    plt.show()

def heatmap_2(w, e, s, n,):
    request = cimgt.OSM()
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection=request.crs))
    extent = [w, e, s, n]  # (xmin, xmax, ymin, ymax)
    ax.set_extent(extent)
    ax.add_image(request, 13)

    # generate (x, y) centering at (extent[0], extent[2])
    x = extent[0] + np.random.randn(1000)
    y = extent[2] + np.random.randn(1000)

    # do coordinate conversion of (x,y)
    xynps = ax.projection.transform_points(ccrs.Geodetic(), x, y)

    # make a 2D histogram
    h = ax.hist2d(xynps[:, 0], xynps[:, 1], bins=40, zorder=10, alpha=0.5)
    # h: (counts, xedges, yedges, image)

    cbar = plt.colorbar(h[3], ax=ax, shrink=0.45, format='%.1f')  # h[3]: image
    plt.show()

def heatmap(time, data, lat, lon):

    # data array
    da = xr.DataArray(data, coords={'time': time, 'lat': lat, 'lon': lon}, dims=['lat', 'lon'])

    # prepare data for annotation
    darray = da
    ylab, xlab = da.dims
    xval = darray[xlab].values
    yval = darray[ylab].values
    zval = darray.to_masked_array(copy=False)
    xval, yval = np.meshgrid(xval, yval)

    # for colormap
    colorlist = generate_cmap(["green", "yellow", "orange", "red", "purple", "maroon"])
    levels = [0, 50, 100, 150, 200, 300, 500]

    # plot
    image = OSM()
    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.BORDERS, edgecolor='gray')

    # ax = plt.axes(projection=image.crs)
    # ax.set_extent([min(lon), max(lon), min(lat), max(lat)])
    # ax.add_image(image, 14)

    da.plot(ax=ax, levels=levels, colors=colorlist, extend='both')

    # for x, y, val in zip(xval.flat, yval.flat, zval.flat):
    #     val = '{}'.format(int(val))
    #     ax.text(x, y, val, ha='center', va='center')

    ax.coastlines(resolution='10m')
    plt.show()

if __name__ == "__main__":

    city = "beijing"
    ll = pd.read_csv("database/city/city_"+city+".csv")
    ll = dict(ll.iloc[0])
    print(ll)

    with open("database/grid/grid_"+city+".csv", "r") as cityfile:
        # lat = set()
        # lon = set()
        # for line in cityfile.readlines()[1:]:
        #     line = line[:-1].split(",")
        #     for i in range(0, 8, 2):
        #         lat.add(float(line[i]))
        #     for i in range(1, 8, 2):
        #         lon.add(float(line[i]))
        # lat = np.array(sorted(list(lat)))
        # lon = np.array(sorted(list(lon), reverse=True))
        lon = 130 + np.arange(20) * 0.5
        lat = 39.0 - np.arange(20) * 0.5
        data = np.random.rand(len(lat), len(lon)) * 100
        time = dt.datetime(2016, 1, 1, 0)
        heatmap(time, data, lat, lon)
        #heatmap_2(-89, -88, 41, 42)
