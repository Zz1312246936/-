import torch
import numpy as np
import torch.nn as nn
from torch.nn import init

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


def conv_block(in_channel, out_channel):  # 一个卷积块
    layer = nn.Sequential(
        nn.BatchNorm1d(in_channel),
        nn.ReLU(),
        nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer


class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super().__init__()  # growth_rate => k => out_channel
        block = []
        channel = in_channel  # channel => in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate  # 连接每层的特征
        self.net = nn.Sequential(*block)  # 实现简单的顺序连接模型
        # 必须确保前一个模块的输出大小和下一个模块的输入大小是一致的

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)  # contact同维度拼接特征，stack(是把list扩维连接
            # torch.cat()是为了把多个tensor进行拼接，在给定维度上对输入的张量序列seq 进行连接操作
            # inputs : 待连接的张量序列，可以是任意相同Tensor类型的python 序列
            # dim : 选择的扩维, 必须在0到len(inputs[0])之间，沿着此维连接张量序列
        return x


def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm1d(in_channel),
        nn.ReLU(),
        nn.Conv1d(in_channel, out_channel, 1),  # kernel_size = 1 1x1 conv
        nn.AvgPool1d(2, 2)  # 2x2 pool
    )
    return trans_layer


class DenseNet(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16]):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channel, 64, 7, 2, 3),  # padding=3 参数要熟悉
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(3, 2, padding=1)
        )
        self.DB1 = self._make_dense_block(64, growth_rate, num=block_layers[0])
        self.TL1 = self._make_transition_layer(256)
        self.DB2 = self._make_dense_block(128, growth_rate, num=block_layers[1])
        self.TL2 = self._make_transition_layer(512)
        self.DB3 = self._make_dense_block(256, growth_rate, num=block_layers[2])
        self.TL3 = self._make_transition_layer(1024)
        self.DB4 = self._make_dense_block(512, growth_rate, num=block_layers[3])
        self.global_avgpool = nn.Sequential(  # 全局平均池化
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d((1, 1))
        )
        self.classifier = nn.Linear(1024, num_classes)  # fc层

    def forward(self, x):
        x = self.block1(x)
        x = self.DB1(x)
        x = self.TL1(x)
        x = self.DB2(x)
        x = self.TL2(x)
        x = self.DB3(x)
        x = self.TL3(x)
        x = self.DB4(x)
        x = self.global_avgpool(x)
        x = self.classifier(x)
        return x

    def _make_dense_block(self, channels, growth_rate, num):  # num是块的个数
        block = []
        block.append(dense_block(channels, growth_rate, num))
        channels += num * growth_rate  # 特征变化 # 这里记录下即可，生成时dense_block()中也做了变化
        return nn.Sequential(*block)

    def _make_transition_layer(self, channels):
        block = []
        block.append(transition(channels, channels // 2))  # channels // 2就是为了降低复杂度 θ = 0.5
        return nn.Sequential(*block)



class My_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(My_LSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout = 0.4)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.selfattention = ScaledDotProductAttention(d_model=128, d_k=128, d_v=128, h=8)
        self.gru_layer = nn.GRU(hidden_size, hidden_size)
        self.dense1 = DenseNet(hidden_size,2)



    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        #gru
        out, _  = self.gru_layer(out)
        out = out.reshape(-1, 128)
        #ATTENTION
        out = self.selfattention(out,out,out)
        #dense
        out = self.dense1(out)
        # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])  # 此处的-1说明我们只取RNN最后输出的那个hn
        # out = self.softmax(out)
        return out

