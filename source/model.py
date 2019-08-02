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

        # drop out
        self.drop_lstm_local = nn.Dropout2d(p=0.3)
        self.drop_lstm_others = nn.Dropout2d(p=0.3)
        self.drop_joint_local = nn.Dropout2d(p=0.2)
        self.drop_joint_others = nn.Dropout2d(p=0.2)
        self.drop_attention = nn.Dropout2d(p=0.2)
        self.drop_fusion = nn.Dropout2d(p=0.1)

        # the neuron size of hidden layer
        self.fc_basic_hidden = 100
        self.fc_joint_hidden = 200
        self.lstm_hidden = 300

        # NN layer
        self.fc_basic = nn.Linear(inputDim_static, self.fc_basic_hidden).to(device)
        # |- local
        self.lstm1_local = nn.LSTM(inputDim_seq_local, self.lstm_hidden, batch_first=True).to(device)
        self.lstm2_local = nn.LSTM(self.lstm_hidden, self.lstm_hidden, batch_first=True).to(device)
        self.fc_joint1_local = nn.Linear(self.fc_basic_hidden + self.lstm_hidden, self.fc_joint_hidden).to(device)
        self.fc_joint2_local = nn.Linear(self.fc_joint_hidden, self.fc_joint_hidden).to(device)

        # |- others
        self.lstm1_others = nn.LSTM(inputDim_seq_others, self.lstm_hidden, batch_first=True).to(device)
        self.lstm2_others = nn.LSTM(self.lstm_hidden, self.lstm_hidden, batch_first=True).to(device)
        self.fc_joint1_others = nn.Linear(self.fc_basic_hidden + self.lstm_hidden, self.fc_joint_hidden).to(device)
        self.fc_joint2_others = nn.Linear(self.fc_joint_hidden, self.fc_joint_hidden).to(device)

        # |- attention
        self.attention1 = nn.Linear(self.fc_joint_hidden + self.fc_joint_hidden, self.fc_joint_hidden + self.fc_joint_hidden).to(device)
        self.attention2 = nn.Linear(self.fc_joint_hidden + self.fc_joint_hidden, 1).to(device)

        # |- fusion
        self.fusion = nn.Linear(self.fc_joint_hidden + self.fc_joint_hidden, self.fc_joint_hidden + self.fc_joint_hidden).to(device)
        self.output = nn.Linear(self.fc_joint_hidden + self.fc_joint_hidden, 1).to(device)


    def forward(self, x_static_local, x_seq_local, x_static_others, x_seq_others, hidden0=None): # Noneは零ベクトルと解釈される
        '''
        :param x_static_local: スタティック特徴量 (local)
        :param x_seq_local: シーケンシャル特徴量 (local)
        :param x_static_others: スタティック特徴量 (others)
        :param x_seq_others: シーケンシャル特徴量 (others)
        :param hidden0: lstm隠れ層の初期値
        :return: aqiの予測値
        '''

        '''
        stage1: extract static and sequence features of local and other stations, respectively
            |- static: 1 layer with 100 neurons (share it with all the stations)
            |- lstm: 2 layers with 300 neurons per layer
        '''

        K = len(x_static_others)

        # |- local
        y_local_static = F.relu(self.fc_basic(x_static_local))
        y_local_static = y_local_static
        y_local_seq, (hidden, cell) = self.lstm1_local(x_seq_local, hidden0)
        y_local_seq = self.drop_lstm_local(y_local_seq)
        y_local_seq, (hidden, cell) = self.lstm2_local(y_local_seq, hidden0)
        y_local_seq = y_local_seq[:, -1, :]
        # |- others
        y_others_static = []
        y_others_seq = []
        for i in range(K):
            y_others_static_i = F.relu(self.fc_basic(x_static_others[i]))
            y_others_static.append(y_others_static_i)
            y_others_seq_i, (hidden, cell) = self.lstm1_others(x_seq_others[i], hidden0)
            y_others_seq_i = self.drop_lstm_others(y_others_seq_i)
            y_others_seq_i, (hidden, cell) = self.lstm2_others(y_others_seq_i, hidden0)
            y_others_seq.append(y_others_seq_i[:, -1, :])

        '''
        stage2: combine static and sequence features on local and other stations, respectively
            |- high-level fc: 2 layers with 200 neurons per layer
        '''
        # |- local
        y_local = F.relu(self.fc_joint1_local(torch.cat([y_local_static, y_local_seq], dim=1)))
        y_local = self.drop_joint_local(y_local)
        y_local = F.relu(self.fc_joint2_local(y_local))
        # |- static
        y_others = []
        for i in range(K):
            y_others_i = F.relu(self.fc_joint1_others(torch.cat([y_others_static[i], y_others_seq[i]], dim=1)))
            y_others_i = self.drop_joint_others(y_others_i)
            y_others_i = F.relu(self.fc_joint2_others(y_others_i))
            y_others.append(y_others_i)

        '''
        stage3: calculate attention for each station by using local and other stations
        '''
        for i in range(K):
            if i == 0:
                attention = F.relu(self.attention1(torch.cat([y_local, y_others[i]], dim=1)))
                attention = self.drop_attention(attention)
                attention = self.attention2(attention)
            else:
                attention_i = F.relu(self.attention1(torch.cat([y_local, y_others[i]], dim=1)))
                attention_i = self.drop_attention(attention_i)
                attention_i = self.attention2(attention_i)
                attention = torch.cat([attention, attention_i], dim=1)

        attention = F.softmax(attention, dim=1)
        attention = torch.chunk(attention, len(y_others), dim=1)
        y_others = [attention[i]*y_others[i] for i in range(len(y_others))]

        '''
        pooling K tensors into 1 tensor 
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
        return self.output(y)
