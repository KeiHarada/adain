from math import *
import pandas as pd
import numpy as np
import overpy
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

class MMD:
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.

    The kernel used is equal to:

    .. math ::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},

    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.

    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        # The three constants used in the test.
        self.a00 = 1. / (n_1 * (n_1 - 1))
        self.a11 = 1. / (n_2 * (n_2 - 1))
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        r"""Evaluate the statistic.

        The kernel used is

        .. math::

            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},

        for the provided ``alphas``.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.

        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(- alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[:self.n_1, :self.n_1]
        k_2 = kernels[self.n_1:, self.n_1:]
        k_12 = kernels[:self.n_1, self.n_1:]

        mmd = (2 * self.a01 * k_12.sum() +
               self.a00 * (k_1.sum() - torch.trace(k_1)) +
               self.a11 * (k_2.sum() - torch.trace(k_2)))
        if ret_matrix:
            return mmd, kernels
        else:
            return mmd

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.local_static = data[0]
        self.local_seq = data[1]
        self.others_static = data[2]
        self.others_seq = data[3]
        self.target = data[4]
        self.data_num = len(data[4])

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):

        out_local_static = torch.tensor(self.local_static[idx])
        out_local_seq = torch.tensor(self.local_seq[idx])
        out_others_static = torch.tensor(self.others_static[idx])
        out_others_seq = torch.tensor(self.others_seq[idx])
        out_target = torch.tensor(self.target[idx])

        return out_local_static, out_local_seq, out_others_static, out_others_seq, out_target

class Color:
    BLACK     = '\033[30m'
    RED       = '\033[31m'
    GREEN     = '\033[32m'
    YELLOW    = '\033[33m'
    BLUE      = '\033[34m'
    PURPLE    = '\033[35m'
    CYAN      = '\033[36m'
    WHITE     = '\033[37m'
    END       = '\033[0m'
    BOLD      = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE   = '\033[07m'

def get_optimizer(trial, model):
    optimizer_names = ['Adam', 'MomentumSGD']
    optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    if optimizer_name == optimizer_names[0]:
        adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
        optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    else:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        optimizer = optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)

    return optimizer

def get_activation(trial):
    activation_names = ['ReLU', 'ELU']
    activation_name = trial.suggest_categorical('activation', activation_names)

    if activation_name == activation_names[0]:
        activation = F.relu
    else:
        activation = F.elu

    return activation

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'\t\tEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'\t\tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        torch.save(model.state_dict(), 'tmp/checkpoint.pt')
        self.val_loss_min = val_loss

def citycode(name, scale):

    codefile = pd.read_csv("rawdata/zheng2015/city.csv", index_col=2, dtype=object)
    code = codefile.at[name, "city_id"]

    if scale == "city":
        return r"%s$" % (code)
    if scale == "district":
        return r"%s\d{2}$" % (code)
    if scale == "station":
        return r"%s\d{3}$" % (code)


    if scale == "station":
        if name == "beijing":
            return r"001\d{3}$"
        if name == "shenzhen":
            return r"004\d{3}$"
        if name == "tianjin":
            return r"006\d{3}$"
        if name == "guangzhou":
            return r"009\d{3}$"

    if scale == "district":
        if name == "beijing":
            return r"001\d{2}$"
        if name == "shenzhen":
            return r"004\d{2}$"
        if name == "tianjin":
            return r"006\d{2}$"
        if name == "guangzhou":
            return r"009\d{2}$"

    if scale == "city":
        if name == "beijing":
            return r"001$"
        if name == "shenzhen":
            return r"004$"
        if name == "tianjin":
            return r"006$"
        if name == "guangzhou":
            return r"009$"

def aqi_class(value):

    if 0.0 <= value < 51.0:
        return "G"
    elif 51.0 <= value < 101.0:
        return "M"
    elif 101.0 <= value < 151.0:
        return "US"
    elif 151.0 <= value < 201.0:
        return "U"
    elif 201.0 <= value < 301.0:
        return "VU"
    elif 301.0 <= value < 501.0:
        return "H"
    else:
        return "ERROR"

def normalization(df):

    scaler = MinMaxScaler([0, 1])
    scaler.fit(df)
    df_n = scaler.transform(df)
    df_n = pd.DataFrame(df_n, columns=df.columns)

    return df_n

def calc_correct(data, label):

    correct = 0
    for i in range(len(data)):
        data_class = aqi_class(data[i])
        label_class = aqi_class(label[i])
        if data_class == label_class:
            correct += 1

    return correct

def calc_winddirection_onehot(value):

    if value == 9:
        idx = 5
    elif value == 13:
        idx = 6
    elif value == 14:
        idx = 7
    elif value == 23:
        idx = 8
    elif value == 24:
        idx = 9
    else:
        idx = value

    onehot = [0]*10
    onehot[idx] = 1
    return onehot

def winddirection_onehot(df):
    df = df.fillna(0.0)
    df = df.astype("int64", copy=False)
    df = df.values
    df = list(map(lambda x: calc_winddirection_onehot(x), df))
    columns = ["wind_direction_"+str(i).zfill(2) for i in range(10)]
    df = pd.DataFrame(df, columns=columns).astype("float", copy=False)
    return df, columns

def weather_onehot(df):
    df = df.fillna(17.0)
    df = df.astype("int64", copy=False)
    df = df.values
    df = list(map(lambda x: calc_weather_onehot(x), df))
    columns = ["weather_"+str(i).zfill(2) for i in range(18)]
    df = pd.DataFrame(df, columns=columns).astype("float", copy=False)
    return df, columns

def calc_weather_onehot(value):
    onehot = [0]*18
    onehot[value] = 1
    return onehot

def ignore_aqi_error(df):
    return df

def data_interpolate(df):
    return df.interpolate(limit_direction='both')

def get_aqi_series(data, sid, attribute):
    return list(data[data["sid"] == sid][attribute])

def get_meteorology_series(data, did, attribute):
    return list(data[data["did"] == did][attribute])

def get_road_data(data, sid, attribute):
    return float(data[data["sid"] == sid][attribute])

def get_poi_data(data, sid, attribute):
    return float(data[data["sid"] == sid][attribute])

def get_road_over_the_city(city):

    with open("database/city/city_" + city + ".csv", "r") as cityfile:
        minlat, minlon, maxlat, maxlon = cityfile.readlines()[1].strip().split(",")
        api = overpy.Overpass()
        #result = api.query("way(" + minlat + "," + minlon + "," + maxlat + "," + maxlon + ");out;")  # (minlat, minlon, maxlat, maxlon)
        result = api.parse_xml(open("rawdata/osm/large/osm_large_"+city).read())
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

    return road


def get_grid_id(city, lat, lon):

    # load data (gid, minlat, minlon, maxlat, maxlon)
    data = open("database/grid/grid_"+city+".csv", "r").readlines()[1:]
    data = list(map(lambda x: x.strip().split(","), data))
    maxlon = sorted(list(set(map(lambda x: float(x[4]), data))))[-1]
    lon_list = sorted(list(set(map(lambda x: float(x[2]), data))))
    lon_list.append(maxlon)
    maxlat = sorted(list(set(map(lambda x: float(x[3]), data))))[-1]
    lat_list = sorted(list(set(map(lambda x: float(x[1]), data))))
    lat_list.append(maxlat)

    mn = 0
    mx = len(lon_list)-1
    while (mx-mn) > 1:
        half = mn+int((mx-mn)/2)
        if lon_list[half] < lon:
            mn = half
        else:
            mx = half
    data = [x for x in data if x[2] == str(lon_list[mn])]
    data = [x for x in data if x[4] == str(lon_list[mx])]

    mn = 0
    mx = len(lat_list)-1
    while (mx-mn) > 1:
        half = mn+int((mx-mn)/2)
        if lat_list[half] < lat:
            mn = half
        else:
            mx = half
    data = [x for x in data if x[1] == str(lat_list[mn])]
    data = [x for x in data if x[3] == str(lat_list[mx])]

    return data[0][0]

def get_dist_angle(lat1, lon1, lat2, lon2, ellipsoid=None):
    '''
    Vincenty法(逆解法)
    2地点の座標(緯度経度)から、距離と方位角を計算する
    :param lat1: 始点の緯度
    :param lon1: 始点の経度
    :param lat2: 終点の緯度
    :param lon2: 終点の経度
    :param ellipsoid: 楕円体
    :return: 距離と方位角
    '''
    
    # 楕円体
    ELLIPSOID_GRS80 = 1  # GRS80
    ELLIPSOID_WGS84 = 2  # WGS84

    # 楕円体ごとの長軸半径と扁平率
    GEODETIC_DATUM = {
        ELLIPSOID_GRS80: [
            6378137.0,  # [GRS80]長軸半径
            1 / 298.257222101,  # [GRS80]扁平率
        ],
        ELLIPSOID_WGS84: [
            6378137.0,  # [WGS84]長軸半径
            1 / 298.257223563,  # [WGS84]扁平率
        ],
    }

    # 反復計算の上限回数
    ITERATION_LIMIT = 1000

    # 差異が無ければ0.0を返す
    if lat1 == lat2 and lon1 == lon2:
        return {
            'distance': 0.0,
            'azimuth1': 0.0,
            'azimuth2': 0.0,
        }

    # 計算時に必要な長軸半径(a)と扁平率(f)を定数から取得し、短軸半径(b)を算出する
    # 楕円体が未指定の場合はGRS80の値を用いる
    a, f = GEODETIC_DATUM.get(ellipsoid, GEODETIC_DATUM.get(ELLIPSOID_GRS80))
    b = (1 - f) * a

    x1 = radians(lat1)
    x2 = radians(lat2)
    y1 = radians(lon1)
    y2 = radians(lon2)

    # 更成緯度(補助球上の緯度)
    U1 = atan((1 - f) * tan(x1))
    U2 = atan((1 - f) * tan(x2))

    sinU1 = sin(U1)
    sinU2 = sin(U2)
    cosU1 = cos(U1)
    cosU2 = cos(U2)

    # 2点間の経度差
    L = y2 - y1

    # wをLで初期化
    w = L

    # 以下の計算をwが収束するまで反復する
    # 地点によっては収束しないことがあり得るため、反復回数に上限を設ける
    for i in range(ITERATION_LIMIT):
        sinw = sin(w)
        cosw = cos(w)
        sinz = sqrt((cosU2 * sinw) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosw) ** 2)
        cosz = sinU1 * sinU2 + cosU1 * cosU2 * cosw
        z = atan2(sinz, cosz)
        sink = cosU1 * cosU2 * sinw / sinz
        cos2k = 1 - sink ** 2
        cos2zm = cosz - 2 * sinU1 * sinU2 / cos2k
        C = f / 16 * cos2k * (4 + f * (4 - 3 * cos2k))
        omega = w
        w = L + (1 - C) * f * sink * (z + C * sinz * (cos2zm + C * cosz * (-1 + 2 * cos2zm ** 2)))

        # 偏差が.000000000001以下ならbreak
        if abs(w - omega) <= 1e-12:
            break
    else:
        # 計算が収束しなかった場合はNoneを返す
        return None

    # wが所望の精度まで収束したら以下の計算を行う
    u2 = cos2k * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    dz = B * sinz * (cos2zm + B / 4 * (cosz * (-1 + 2 * cos2zm ** 2) - B / 6 * cos2zm * (-3 + 4 * sinz ** 2) * (-3 + 4 * cos2zm ** 2)))

    # 2点間の楕円体上の距離
    s = b * A * (z - dz)

    # 各点における方位角
    k1 = atan2(cosU2 * sinw, cosU1 * sinU2 - sinU1 * cosU2 * cosw)
    k2 = atan2(cosU1 * sinw, -sinU1 * cosU2 + cosU1 * sinU2 * cosw) + pi

    if (k1 < 0):
        k1 = k1 + pi * 2.0

    return {
        'distance': s,           # 距離
        'azimuth1': degrees(k1), # 方位角(始点→終点)
        'azimuth2': degrees(k2), # 方位角(終点→始点)
    }