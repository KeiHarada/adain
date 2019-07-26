import requests
import io
import string
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm

from scipy import misc

from mpl_toolkits.basemap import Basemap
import gdal

from math import pi
from math import tanh
from math import sin
from math import asin
from math import exp
from numpy import arctanh


# ピクセル座標を緯度経度に変換する
def fromPointToLatLng(pixelLat, pixelLon, z):
    L = 85.05112878
    lon = 180 * ((pixelLon / 2.0 ** (z + 7)) - 1)
    lat = 180 / pi * (asin(tanh(-pi / 2 ** (z + 7) * pixelLat + arctanh(sin(pi / 180 * L)))))
    return lat, lon


# 地形データを読み込む
def load_gis(urlFormat, z, x1, x2, y1, y2):
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):

            # 地形データを読み込む
            url = urlFormat.format(z, x, y)
            print
            url
            response = requests.get(url)
            if response.status_code == 404:
                Z = np.zeros((256, 256))
            else:
                # 標高値がない区画はとりあえず0mに置換する
                maptxt = string.replace(response.text, u'e', u'0.0')
                Z = pd.read_csv(io.StringIO(maptxt), header=None)
                Z = Z.values

            # 　標高タイルを縦に連結
            if y == y1:
                gis_v = Z
            else:
                # gis_v = cv2.vconcat([gis_v, Z])
                # gis_v = np.append(gis_v,Z,0)
                gis_v = np.concatenate((gis_v, Z), axis=0)  # 縦

        # 標高タイルを横に連結
        if x == x1:
            gis = gis_v
        else:
            # gis = cv2.hconcat([gis, gis_v])
            # gis = np.append(gis,gis_v,1)
            gis = np.concatenate((gis, gis_v), axis=1)  # 横

    return gis


# 地図画像データを読み込み各ピクセルをRGB値に変換した配列を返す
def load_imgColors(urlFormat, z, x1, x2, y1, y2):
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):

            # 地図画像データを読み込む
            url = urlFormat.format(z, x, y)
            print
            url
            response = requests.get(url)
            if response.status_code == 404:

                # 地図画像データが無い区画は白塗りにする
                colors = np.ones((256, 256, 3), dtype=object)

            else:

                # 画像READ
                img = misc.imread(StringIO(response.content))

            # 　画像タイルを縦に連結
            if y == y1:
                im_v = img
            else:
                # im_v = cv2.vconcat([im_v, img])
                # im_v = np.append(im_v,img,0)
                im_v = np.concatenate((im_v, img), axis=0)  # 縦

        # 画像タイルを横に連結
        if x == x1:
            im = im_v
        else:
            # im = cv2.hconcat([im, im_v])
            # im = np.append(im,im_v,1)
            im = np.concatenate((im, im_v), axis=1)  # 横

    return im


z = 5
x1 = 27
x2 = 29
y1 = 11
y2 = 13

# ズームレベルに応じた経緯度線のステップ
coordinateLineStep = {
    5: 10.0,
    6: 10.0,
    7: 5.0,
    8: 2.5,
    9: 1.0,
    10: 0.5,
    11: 0.25,
    12: 0.2,
    13: 0.1,
    14: 0.05,
    15: 0.02
}

# 標高タイルのURLフォーマット
urlFormat = 'http://cyberjapandata.gsi.go.jp/xyz/dem/{0}/{1}/{2}.txt'

# 標高タイルを読み込み連結して１枚の標高タイルとして返す
Z = load_gis(urlFormat, z, x1, x2, y1, y2)

# OpenStreetMapのタイル画像のURLフォーマット
urlFormat = 'http://a.tile.openstreetmap.org/{0}/{1}/{2}.png'

# OpenStreetMapのタイル画像を読み込み連結して１枚の画像として返す
imgColors = load_imgColors(urlFormat, z, x1, x2, y1, y2)

# 地図画像RGBデータは256で割って0..1に正規化しておく
imgColors = imgColors / 256.

# 勾配を求める
(Zy, Zx) = np.gradient(Z)

# Y軸方向の勾配を0..1に正規化
Zgradient_norm = (Zy - Zy.min()) / (Zy.max() - Zy.min())

# Y軸方向の勾配をグレイスケール化する
projectionIMG = cm.binary(Zgradient_norm)

# 透過情報はカット
projectionIMG = projectionIMG[:, :, :3]

# 地図画像と射影印影図を足して２で割りゃ画像が合成される
imgColors = (imgColors * 1.2 + projectionIMG) / 2

# 合成画像の輝度値の標準偏差を32,平均を80になんとなく変更
imgColors = (imgColors - np.mean(imgColors)) / np.std(imgColors) * 32 + 80
imgColors = (imgColors - imgColors.min()) / (imgColors.max() - imgColors.min())

# 地図の描画範囲の緯度経度を求める
xlim = Z.shape[1]
ylim = Z.shape[0]

minLat, minLon = fromPointToLatLng(y1 * 256 + ylim, x1 * 256, z)
maxLat, maxLon = fromPointToLatLng(y1 * 256, x1 * 256 + xlim, z)

# 地図を作成する
fig = plt.figure(figsize=(12, 9.8))
map = Basemap(epsg=3857,
              lat_ts=38, resolution='h',
              llcrnrlon=minLon, llcrnrlat=minLat,
              urcrnrlon=maxLon, urcrnrlat=maxLat)

# 海岸線を描画
map.drawcoastlines(linewidth=1.0, color='k')

# 経緯度線を描画
map.drawparallels(np.arange(10.0, 120.0, coordinateLineStep[z]), labels=[1, 0, 0, 0], fontsize=18)
map.drawmeridians(np.arange(50.0, 180.0, coordinateLineStep[z]), labels=[0, 0, 0, 1], fontsize=18)

# 背景地図画像を描画
mx0, my0 = map(minLon, minLat)
mx1, my1 = map(maxLon, maxLat)

extent = (mx0, mx1, my0, my1)
plt.imshow(imgColors, vmin=0, vmax=255, extent=extent)

# X,Y軸のグリッドを生成
# x = [fromPointToLatLng(y1* 256, x1* 256 + i, z)[1] for i in range(0, xlim)]
# y = [fromPointToLatLng(y1* 256 + i, 0, z)[0] for i in range(ylim, 0, -1)]

# x, y = map(x, y)

# x = np.linspace(0, map.urcrnrx, Z.shape[1])
# y = np.linspace(0, map.urcrnry, Z.shape[0])

# X, Y = np.meshgrid(x, y)

# 標高250m間隔で等高線を描く
# elevation = range(0,4000,250)
# cont = map.contour(X, Y, Z, levels=elevation,linewidth=1, linestyles = 'solid', cmap='PiYG')
# cont = map.contour(X, Y, Z, levels=elevation,linewidth=1, colors = '#c0d0d0')
# cont = map.contour(X, Y, Z, levels=elevation,linewidth=1, colors = 'k', linestyles = 'solid')
# cont.clabel(fmt='%1.1fm', fontsize=6)


plt.savefig('1.jpg', dpi=72)
plt.show()