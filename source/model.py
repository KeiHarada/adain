import torch
import torch.nn as nn
import torch.nn.functional as F

# FNN
class FNN(nn.Module):
    def __init__(self, inputDim):
        super(FNN, self).__init__()

        # CPU or GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # drop out layer
        self.drop = nn.Dropout2d(p=0.3)

        # the neuron size of hidden layer
        self.hidden = 200

        # NN layer
        self.fc1 = nn.Linear(inputDim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.hidden)
        self.output = nn.Linear(self.hidden, 1)
        nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc3.weight, -0.1, 0.1)
        nn.init.uniform_(self.output.weight, -0.1, 0.1)
        self.fc1.to(device)
        self.fc2.to(device)
        self.fc3.to(device)
        self.output.to(device)

    def forward(self, x):

        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.drop(y)
        y = F.relu(self.fc3(y))
        return self.output(y)

# ADAIN
class ADAIN(nn.Module):

    def __init__(self, inputDim_local_static, inputDim_local_seq, inputDim_others_static, inputDim_others_seq):
        super(ADAIN, self).__init__()

        # CPU or GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # drop out layer
        self.drop_static_local = nn.Dropout2d(p=0.3)
        self.drop_static_others = nn.Dropout2d(p=0.3)
        self.drop_lstm_local = nn.Dropout2d(p=0.3)
        self.drop_lstm_others = nn.Dropout2d(p=0.3)
        self.drop_joint_local = nn.Dropout2d(p=0.2)
        self.drop_joint_others = nn.Dropout2d(p=0.2)
        self.drop_attention = nn.Dropout2d(p=0.1)
        self.drop_fusion = nn.Dropout2d(p=0.1)

        # the neuron size of hidden layer
        self.fc_static_hidden = 100
        self.fc_joint_hidden = 200
        self.lstm_hidden = 300

        # NN layer
        # |- local
        self.fc_static_local = nn.Linear(inputDim_local_static, self.fc_static_hidden)
        self.lstm1_local = nn.LSTM(inputDim_local_seq, self.lstm_hidden, batch_first=True).to(device)
        self.lstm2_local = nn.LSTM(self.lstm_hidden, self.lstm_hidden, batch_first=True).to(device)
        self.fc_joint1_local = nn.Linear(self.fc_static_hidden + self.lstm_hidden, self.fc_joint_hidden)
        self.fc_joint2_local = nn.Linear(self.fc_joint_hidden, self.fc_joint_hidden)
        nn.init.uniform_(self.fc_static_local.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_joint1_local.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_joint2_local.weight, -0.1, 0.1)
        self.fc_static_local.to(device)
        self.fc_joint1_local.to(device)
        self.fc_joint2_local.to(device)

        # |- others
        self.fc_static_others = nn.Linear(inputDim_others_static, self.fc_static_hidden)
        self.lstm1_others = nn.LSTM(inputDim_others_seq, self.lstm_hidden, batch_first=True).to(device)
        self.lstm2_others = nn.LSTM(self.lstm_hidden, self.lstm_hidden, batch_first=True).to(device)
        self.fc_joint1_others = nn.Linear(self.fc_static_hidden + self.lstm_hidden, self.fc_joint_hidden)
        self.fc_joint2_others = nn.Linear(self.fc_joint_hidden, self.fc_joint_hidden)
        nn.init.uniform_(self.fc_static_others.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_joint1_others.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_joint2_others.weight, -0.1, 0.1)
        self.fc_static_others.to(device)
        self.fc_joint1_others.to(device)
        self.fc_joint2_others.to(device)

        # |- attention
        self.attention1 = nn.Linear(self.fc_joint_hidden + self.fc_joint_hidden, self.fc_joint_hidden + self.fc_joint_hidden)
        self.attention2 = nn.Linear(self.fc_joint_hidden + self.fc_joint_hidden, 1)
        nn.init.uniform_(self.attention1.weight, -0.1, 0.1)
        nn.init.uniform_(self.attention2.weight, -0.1, 0.1)
        self.attention1.to(device)
        self.attention2.to(device)

        # |- fusion
        self.fusion = nn.Linear(self.fc_joint_hidden + self.fc_joint_hidden, self.fc_joint_hidden + self.fc_joint_hidden)
        self.output = nn.Linear(self.fc_joint_hidden + self.fc_joint_hidden, 1)
        nn.init.uniform_(self.fusion.weight, -0.1, 0.1)
        nn.init.uniform_(self.output.weight, -0.1, 0.1)
        self.fusion.to(device)
        self.output.to(device)

    def initial_hidden(self, data_size):
        '''
        :param data_size:
        :param batch_size:
        :return: (hidden0, cell0)
        '''

        # CPU or GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"

        hidden0 = ((torch.rand(1, data_size, self.lstm_hidden)-0.5)*0.2).to(device)
        cell0 = ((torch.rand(1, data_size, self.lstm_hidden)-0.5)*0.2).to(device)

        return (hidden0, cell0)

    def forward(self, x_local_static, x_local_seq, x_others_static, x_others_seq):
        '''
        :param x_local_static: スタティック特徴量 (local)
        :param x_local_seq: シーケンシャル特徴量 (local)
        :param x_others_static: スタティック特徴量 (others)
        :param x_others_seq: シーケンシャル特徴量 (others)
        :param hidden0: lstm隠れ層の初期値
        :return: aqiの予測値
        '''

        '''
        stage1: extract static and sequence features of local and other stations, respectively
            |- static: 1 layer with 100 neurons (share it with all the stations)
            |- lstm: 2 layers with 300 neurons per layer
        stage2: combine static and sequence features on local and other stations, respectively
            |- high-level fc: 2 layers with 200 neurons per layer
        stage3: calculate attention for each station by using local and other stations
            |- attention fc: MLP layer
        stage4: fusion local and others
            |- fusion fc: MLP layer
        '''

        # forwarding
        # |- local
        # basic layer
        y_local_static = F.relu(self.fc_static_local(x_local_static))
        y_local_static = self.drop_static_local(y_local_static)

        # lstm layer
        y_local_seq, (hidden, cell) = self.lstm1_local(x_local_seq, self.initial_hidden(len(x_local_seq)))
        y_local_seq, (hidden, cell) = self.lstm2_local(y_local_seq, self.initial_hidden(len(x_local_seq)))
        y_local_seq = self.drop_lstm_local(hidden[0]) # 最後の出力が欲しいのでhiddenを使う

        # joint layer
        y_local = F.relu(self.fc_joint1_local(torch.cat([y_local_static, y_local_seq], dim=1)))
        y_local = F.relu(self.fc_joint2_local(y_local))
        y_local = self.drop_joint_local(y_local)

        # |- others
        # the number of other stations
        K = x_others_static.size(dim=1)
        x_others_static = [torch.squeeze(x) for x in torch.chunk(x_others_static, K, dim=1)]
        x_others_seq = [torch.squeeze(x) for x in torch.chunk(x_others_seq, K, dim=1)]
        y_others = list()
        attention = list()
        for i in range(K):
            # basic layer
            y_others_static_i = F.relu(self.fc_static_others(x_others_static[i]))
            y_others_static_i = self.drop_static_others(y_others_static_i)

            # lstm layer
            y_others_seq_i, (hidden, cell) = self.lstm1_others(x_others_seq[i], self.initial_hidden(len(x_local_seq)))
            y_others_seq_i, (hidden, cell) = self.lstm2_others(y_others_seq_i, self.initial_hidden(len(x_local_seq)))
            y_others_seq_i = self.drop_lstm_others(hidden[0])  # 最後の出力が欲しいのでhiddenを使う

            # joint layer
            y_others_i = F.relu(self.fc_joint1_others(torch.cat([y_others_static_i, y_others_seq_i], dim=1)))
            y_others_i = F.relu(self.fc_joint2_others(y_others_i))
            y_others_i = self.drop_joint_others(y_others_i)
            y_others.append(y_others_i)

            # attention layer
            attention_i = F.relu(self.attention1(torch.cat([y_local, y_others_i], dim=1)))
            attention_i = self.drop_attention(attention_i)
            attention_i = self.attention2(attention_i)
            attention.append(attention_i)

        # give other stations attention score
        y_others = torch.stack(y_others, dim=0)
        attention = torch.stack(attention, dim=0)
        attention = F.softmax(attention, dim=0)
        y_others = (attention * y_others).sum(dim=0)

        # output layer
        y = F.relu(self.fusion(torch.cat([y_local, y_others], dim=1)))
        y = self.drop_fusion(y)
        y = self.output(y)
        return y

# HARADA
class HARADA(nn.Module):

    def __init__(self, inputDim_local_static, inputDim_local_seq, inputDim_others_static, inputDim_others_seq, cityNum, stationNum):
        super(HARADA, self).__init__()

        # CPU or GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cityNum = cityNum
        self.stationNum = stationNum
        self.lstm_num_layers = 2

        # drop out layer
        self.drop_static_local = nn.Dropout(p=0.5)
        self.drop_static_others = nn.Dropout(p=0.5)
        self.drop_joint_local = nn.Dropout(p=0.5)
        self.drop_joint_others = nn.Dropout(p=0.5)
        self.drop_attention_station = nn.Dropout(p=0.5)
        self.drop_attention_city = nn.Dropout(p=0.5)
        self.drop_fusion = nn.Dropout(p=0.5)

        # the neuron size of hidden layer
        self.fc_static_hidden = 300
        self.fc_joint_hidden = 300
        self.fc_attention_station_hidden = self.fc_joint_hidden * 2
        self.fc_attention_city_hidden = (self.fc_joint_hidden * (self.stationNum + 1)) + 2
        self.fusion_hidden = 300
        self.lstm_hidden = 300

        # NN layer
        # |- local
        self.fc_static_local1 = nn.Linear(inputDim_local_static, self.fc_static_hidden)
        self.fc_static_local2 = nn.Linear(self.fc_static_hidden, self.fc_static_hidden)
        self.lstm_local = nn.LSTM(inputDim_local_seq, self.lstm_hidden, batch_first=True, dropout=0.2, num_layers=self.lstm_num_layers).to(device)
        self.fc_joint_local1 = nn.Linear(self.fc_static_hidden + self.lstm_hidden, self.fc_joint_hidden)
        self.fc_joint_local2 = nn.Linear(self.fc_joint_hidden, self.fc_joint_hidden)
        nn.init.uniform_(self.fc_static_local1.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_static_local2.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_joint_local1.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_joint_local2.weight, -0.1, 0.1)
        self.fc_static_local1.to(device)
        self.fc_static_local2.to(device)
        self.fc_joint_local1.to(device)
        self.fc_joint_local2.to(device)

        # |- others
        self.fc_static_others1 = nn.Linear(inputDim_others_static, self.fc_static_hidden)
        self.fc_static_others2 = nn.Linear(self.fc_static_hidden, self.fc_static_hidden)
        self.lstm_others = nn.LSTM(inputDim_others_seq, self.lstm_hidden, batch_first=True, dropout=0.2, num_layers=self.lstm_num_layers).to(device)
        self.fc_joint_others1 = nn.Linear(self.fc_static_hidden + self.lstm_hidden, self.fc_joint_hidden)
        self.fc_joint_others2 = nn.Linear(self.fc_joint_hidden, self.fc_joint_hidden)
        nn.init.uniform_(self.fc_static_others1.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_static_others2.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_joint_others1.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_joint_others2.weight, -0.1, 0.1)
        self.fc_static_others1.to(device)
        self.fc_static_others2.to(device)
        self.fc_joint_others1.to(device)
        self.fc_joint_others2.to(device)

        # |- attention
        self.attention_station1 = nn.Linear(self.fc_attention_station_hidden, self.fc_attention_station_hidden)
        self.attention_station2 = nn.Linear(self.fc_attention_station_hidden, 1)
        self.attention_city1 = nn.Linear(self.fc_attention_city_hidden, self.fc_attention_city_hidden)
        self.attention_city2 = nn.Linear(self.fc_attention_city_hidden, 1)
        nn.init.uniform_(self.attention_station1.weight, -0.1, 0.1)
        nn.init.uniform_(self.attention_station2.weight, -0.1, 0.1)
        nn.init.uniform_(self.attention_city1.weight, -0.1, 0.1)
        nn.init.uniform_(self.attention_city2.weight, -0.1, 0.1)
        self.attention_station1.to(device)
        self.attention_station2.to(device)
        self.attention_city1.to(device)
        self.attention_city2.to(device)

        # |- output
        self.fusion1 = nn.Linear(self.fc_attention_station_hidden, self.fusion_hidden)
        self.fusion2 = nn.Linear(self.fusion_hidden, self.fusion_hidden)
        self.output = nn.Linear(self.fusion_hidden, 1)
        nn.init.uniform_(self.fusion1.weight, -0.1, 0.1)
        nn.init.uniform_(self.fusion2.weight, -0.1, 0.1)
        nn.init.uniform_(self.output.weight, -0.1, 0.1)
        self.fusion1.to(device)
        self.fusion2.to(device)
        self.output.to(device)


    def initial_hidden(self, data_size):
        '''
        :param data_size:
        :param batch_size:
        :return: (hidden0, cell0)
        '''

        # CPU or GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"

        hidden0 = ((torch.rand(self.lstm_num_layers, data_size, self.lstm_hidden)-0.5)*0.2).to(device)
        cell0 = ((torch.rand(self.lstm_num_layers, data_size, self.lstm_hidden)-0.5)*0.2).to(device)

        return (hidden0, cell0)

    def encode(self, x_local_static, x_local_seq):
        # static layer
        y_local_static = F.relu(self.fc_static_local1(x_local_static))
        y_local_static = self.drop_static_local(y_local_static)
        y_local_static = F.relu(self.fc_static_local2(y_local_static))

        # lstm layer
        y_local_seq, (hidden, cell) = self.lstm_local(x_local_seq, self.initial_hidden(len(x_local_seq)))
        y_local_seq = hidden[0] # 最後の出力が欲しいのでhiddenを使う

        # joint layer
        y_local = F.relu(self.fc_joint_local1(torch.cat([y_local_static, y_local_seq], dim=1)))
        y_local = self.drop_joint_local(y_local)
        y_local = F.relu(self.fc_joint_local2(y_local))

        return y_local

    def forward(self, x_local_static, x_local_seq, x_others_static, x_others_seq, x_others_city):
        '''
        :param x_local_static: スタティック特徴量 (local)
        :param x_local_seq: シーケンシャル特徴量 (local)
        :param x_others_static: スタティック特徴量 (others)
        :param x_others_seq: シーケンシャル特徴量 (others)
        :param x_others_city: 都市間の距離と角度
        :return: aqiの予測値
        '''

        # output
        y_mtl = list()
        mmd = list()

        # |- local
        # static layer
        y_local_static = F.relu(self.fc_static_local1(x_local_static))
        y_local_static = self.drop_static_local(y_local_static)
        y_local_static = F.relu(self.fc_static_local2(y_local_static))

        # lstm layer
        y_local_seq, (hidden, cell) = self.lstm_local(x_local_seq, self.initial_hidden(len(x_local_seq)))
        y_local_seq = hidden[0] # 最後の出力が欲しいのでhiddenを使う

        # joint layer
        y_local = F.relu(self.fc_joint_local1(torch.cat([y_local_static, y_local_seq], dim=1)))
        y_local = self.drop_joint_local(y_local)
        y_local = F.relu(self.fc_joint_local2(y_local))

        # |- others
        # slicing by the number of cities
        x_others_static = [torch.squeeze(x) for x in torch.chunk(x_others_static, self.cityNum, dim=1)]
        x_others_seq = [torch.squeeze(x) for x in torch.chunk(x_others_seq, self.cityNum, dim=1)]
        x_others_city = [torch.squeeze(x) for x in torch.chunk(x_others_city, self.cityNum, dim=1)]

        attention_city = list()
        for i in range(self.cityNum):

            y_others_i = list()
            attention_station_i = list()

            # slicing by the number of stations
            x_others_static_i = [torch.squeeze(x) for x in torch.chunk(x_others_static[i], self.stationNum, dim=1)]
            x_others_seq_i = [torch.squeeze(x) for x in torch.chunk(x_others_seq[i], self.stationNum, dim=1)]

            for j in range(self.stationNum):

                mmd.append(self.encode(x_others_static_i[j][:, :-2], x_others_seq_i[j][:, :, :-1]))

                # basic layer
                y_others_static_ij = F.relu(self.fc_static_others1(x_others_static_i[j]))
                y_others_static_ij = self.drop_static_others(y_others_static_ij)
                y_others_static_ij = F.relu(self.fc_static_others2(y_others_static_ij))

                # lstm layer
                y_others_seq_ij, (hidden, cell) = self.lstm_others(x_others_seq_i[j], self.initial_hidden(len(x_local_seq)))
                y_others_seq_ij = hidden[0]  # 最後の出力が欲しいのでhiddenを使う

                # joint layer
                y_others_ij = F.relu(self.fc_joint_others1(torch.cat([y_others_static_ij, y_others_seq_ij], dim=1)))
                y_others_ij = self.drop_joint_others(y_others_ij)
                y_others_ij = F.relu(self.fc_joint_others2(y_others_ij))
                y_others_i.append(y_others_ij)

                # station based attention layer
                attention_station_ij = F.relu(self.attention_station1(torch.cat([y_local, y_others_ij], dim=1)))
                attention_station_ij = self.drop_attention_station(attention_station_ij)
                attention_station_ij = self.attention_station2(attention_station_ij)
                attention_station_i.append(attention_station_ij)

            # joint layer
            y_city_i = torch.cat(y_others_i, dim=1)
            y_city_i = torch.cat([y_city_i, x_others_city[i]], dim=1)

            # city based attention layer
            attention_city_i = F.relu(self.attention_city1(torch.cat([y_local, y_city_i], dim=1)))
            attention_city_i = self.drop_attention_city(attention_city_i)
            attention_city_i = self.attention_city2(attention_city_i)
            attention_city.append(attention_city_i)

            # give other stations attention score
            y_others_i = torch.stack(y_others_i, dim=0)
            attention_station_i = torch.stack(attention_station_i, dim=0)
            attention_station_i = F.softmax(attention_station_i, dim=0)
            y_others_i = (attention_station_i * y_others_i).sum(dim=0)

            # output layer
            y_i = F.relu(self.fusion1(torch.cat([y_local, y_others_i], dim=1)))
            y_i = self.drop_fusion(y_i)
            y_i = F.relu(self.fusion2(y_i))
            y_i = self.output(y_i)
            if self.training is False:
                print("\t\t", y_i[0])
            y_mtl.append(y_i)

        if self.training is False:
            print("\t\t---")

        # output
        y_mmd = torch.cat(mmd, dim=0)

        # give other cities attention score
        y_moe = torch.stack(y_mtl, dim=0)
        attention_city = torch.stack(attention_city, dim=0)
        attention_city = F.softmax(attention_city, dim=0)
        y_moe = (attention_city * y_moe).sum(dim=0)

        etp = -1.0 * (attention_city * torch.log(attention_city + 1e-7)).sum(dim=0)
        etp = torch.sum(etp)

        return y_moe, y_mtl, y_mmd, etp