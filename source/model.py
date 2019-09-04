import torch
import torch.nn as nn
import torch.nn.functional as F

# ADAIN
class ADAIN(nn.Module):

    def __init__(self, inputDim_static, inputDim_seq_local, inputDim_seq_others):
        super(ADAIN, self).__init__()

        # CPU or GPU
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # drop out layer
        self.drop_basic = nn.Dropout(p=0.3)
        self.drop_lstm_local = nn.Dropout(p=0.3)
        self.drop_lstm_others = nn.Dropout(p=0.3)
        self.drop_joint_local = nn.Dropout(p=0.2)
        self.drop_joint_others = nn.Dropout(p=0.2)
        self.drop_fusion = nn.Dropout(p=0.1)

        # the neuron size of hidden layer
        self.fc_basic_hidden = 100
        self.fc_joint_hidden = 200
        self.lstm_hidden = 300

        # NN layer
        self.fc_basic = nn.Linear(inputDim_static, self.fc_basic_hidden)
        nn.init.uniform_(self.fc_basic.weight, -0.1, 0.1)
        self.fc_basic.to(device)
        # |- local
        self.lstm1_local = nn.LSTM(inputDim_seq_local, self.lstm_hidden, batch_first=True).to(device)
        self.lstm2_local = nn.LSTM(self.lstm_hidden, self.lstm_hidden, batch_first=True).to(device)
        self.fc_joint1_local = nn.Linear(self.fc_basic_hidden + self.lstm_hidden, self.fc_joint_hidden)
        self.fc_joint2_local = nn.Linear(self.fc_joint_hidden, self.fc_joint_hidden)
        nn.init.uniform_(self.fc_joint1_local.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_joint2_local.weight, -0.1, 0.1)
        self.fc_joint1_local.to(device)
        self.fc_joint2_local.to(device)

        # |- others
        self.lstm1_others = nn.LSTM(inputDim_seq_others, self.lstm_hidden, batch_first=True).to(device)
        self.lstm2_others = nn.LSTM(self.lstm_hidden, self.lstm_hidden, batch_first=True).to(device)
        self.fc_joint1_others = nn.Linear(self.fc_basic_hidden + self.lstm_hidden, self.fc_joint_hidden)
        self.fc_joint2_others = nn.Linear(self.fc_joint_hidden, self.fc_joint_hidden)
        nn.init.uniform_(self.fc_joint1_others.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_joint2_others.weight, -0.1, 0.1)
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
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

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
        '''

        # the number of other stations
        K = len(x_others_static)

        # |- local
        y_local_static = F.relu(self.fc_basic(x_local_static))
        y_local_static = self.drop_basic(y_local_static)
        y_local_seq, (hidden, cell) = self.lstm1_local(x_local_seq, self.initial_hidden(len(x_local_seq)))
        y_local_seq, (hidden, cell) = self.lstm2_local(y_local_seq, self.initial_hidden(len(x_local_seq)))
        y_local_seq = self.drop_lstm_local(hidden[0]) # 最後の出力が欲しいのでhiddenを使う
        # |- others
        y_others_static = []
        y_others_seq = []
        for i in range(K):
            y_others_static_i = F.relu(self.fc_basic(x_others_static[i]))
            y_others_static_i = self.drop_basic(y_others_static_i)
            y_others_static.append(y_others_static_i)
            y_others_seq_i, (hidden, cell) = self.lstm1_others(x_others_seq[i], self.initial_hidden(len(x_local_seq)))
            y_others_seq_i, (hidden, cell) = self.lstm2_others(y_others_seq_i, self.initial_hidden(len(x_local_seq)))
            y_others_seq_i = self.drop_lstm_others(hidden[0]) # 最後の出力が欲しいのでhiddenを使う
            y_others_seq.append(y_others_seq_i)

        '''
        stage2: combine static and sequence features on local and other stations, respectively
            |- high-level fc: 2 layers with 200 neurons per layer
        '''
        # |- local
        y_local = F.relu(self.fc_joint1_local(torch.cat([y_local_static, y_local_seq], dim=1)))
        y_local = F.relu(self.fc_joint2_local(y_local))
        y_local = self.drop_joint_local(y_local)
        # |- others
        y_others = []
        for i in range(K):
            y_others_i = F.relu(self.fc_joint1_others(torch.cat([y_others_static[i], y_others_seq[i]], dim=1)))
            y_others_i = F.relu(self.fc_joint2_others(y_others_i))
            y_others_i = self.drop_joint_others(y_others_i)
            y_others.append(y_others_i)

        '''
        stage3: calculate attention for each station by using local and other stations
        '''
        for i in range(K):
            if i == 0:
                attention = F.relu(self.attention1(torch.cat([y_local, y_others[i]], dim=1)))
                attention = self.attention2(attention)
            else:
                attention_i = F.relu(self.attention1(torch.cat([y_local, y_others[i]], dim=1)))
                attention_i = self.attention2(attention_i)
                attention = torch.cat([attention, attention_i], dim=1)

        attention = F.softmax(attention, dim=1) # attention score vector
        attention = torch.chunk(attention, len(y_others), dim=1) # split attention score vector
        y_others = [attention[i]*y_others[i] for i in range(len(y_others))] # give y_others attention

        '''
        sum K tensors 
        '''
        tmp = y_others[0]
        for i in y_others[1:]:
            tmp += i
        y_others = tmp

        '''
        stage4: fusion local and attention 
        '''
        y = F.relu(self.fusion(torch.cat([y_local, y_others], dim=1)))
        y = self.drop_fusion(y)
        y = self.output(y)
        return y
