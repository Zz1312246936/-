
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os


import numpy
import pandas as pd
import numpy as np
from scipy.stats import entropy, kurtosis, skew
from scipy.signal import find_peaks
from scipy import signal
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from My_LSTM import My_LSTM

# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers=5, batch_first=True, dropout = 0.4)
#         self.fc = nn.Linear(hidden_size, num_classes)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         # Set initial hidden and cell states
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#
#         # Forward propagate LSTM
#         out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
#
#         # Decode the hidden state of the last time step
#         out = self.fc(out[:, -1, :])  # 此处的-1说明我们只取RNN最后输出的那个hn
#         # out = self.softmax(out)
#         return out
#
#



class myDataSet(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        """
        :param data_dir: 数据文件路径
        :param label_dir: 标签文件路径
        :param transform: transform操作
        """
        self.transform = transform
        # 读文件夹下每个数据文件名称
        # os.listdir读取文件夹内的文件名称
        self.file_name = os.listdir(data_dir)

        self.data_path = []
        self.label_path = label_dir
        self.labels = pd.read_csv(self.label_path, header=None, skiprows=1)

        # 让每一个文件的路径拼接起来
        for index in range(len(self.file_name)):
            self.data_path.append(os.path.join(data_dir, self.file_name[index]))
            # self.label_path.append(os.path.join(label_dir, self.label_name[index]))

    def __len__(self):
        # 返回数据集长度
        return len(self.file_name)

    def __getitem__(self, index):
        # 获取每一个数据

        # 读取数据
        data = pd.read_csv(self.data_path[index], header=None, skiprows=1)
        # 读取标签

        # label = label[index, 3]
        label = self.labels.iloc[index, 3]
        if self.transform:
            data = self.transform(data)
            label = self.transform(label)

        # 转成张量
        numpy_array = data.values
        data = torch.tensor(numpy_array, dtype=torch.float32)
        # data = torch.tensor(data.values)
        label = torch.tensor(label)

        return index, data, label  # 返回数据和标签
# 测试函数
if __name__ == "__main__":
    data_dir = 'F:\\rnn训练\\训练集'
    label_dir = 'F:\\rnn训练\\训练标签\\metadata_train.csv'
    data_dir1 = 'F:\\rnn训练\\测试集'
    label_dir1 = 'F:\\rnn训练\\测试标签\\metadata_train.csv'
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # Hyper-parameters
    sequence_length = 100
    input_size = 31
    hidden_size = 128
    num_layers = 1
    num_classes = 2
    batch_size = 100
    num_epochs = 10
    learning_rate = 1e-4

    # 读取数据集
    train_dataset = myDataSet(
        data_dir=data_dir,
        label_dir=label_dir,
    )
    test_dataset = myDataSet(
        data_dir=data_dir1,
        label_dir=label_dir1,
    )
    # 加载数据集
    train_iter = DataLoader(train_dataset)
    test_iter = DataLoader(test_dataset)
    # for index in range(8712):
    #     # 构建输入文件名
    #     input_file = f'features_{index}.csv'
    #     input_folder = 'F:\归一化测试'  # 输出CSV文件夹路径
    #     input_file_path = os.path.join(input_folder, input_file)
    #     data = pd.read_csv(input_file_path, header=None, skiprows=1).values.astype(float)

    model = My_LSTM(input_size, hidden_size, num_layers, num_classes, device).to(device)
    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    weights = torch.tensor([0.5, 20]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_dataset)
    lossum = 0
    for epoch in range(num_epochs):
        for index, data, label in train_iter:
            data = data.reshape(-1, sequence_length, input_size).to(device)
            label = label.to(device)
            # weights = torch.tensor([14.3, 1.1])
            # Forward pass
            outputs = model(data)
            # loss = criterion(outputs, label)
            loss = criterion(outputs, label)
            lossum += loss.detach()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if index % 1000 == 5:
                print(epoch, index, outputs.cpu().detach().numpy(), label.cpu().detach().numpy(),loss.cpu().detach().numpy() )

        print('Epoch [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, lossum.item()))
        lossum = 0
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for index, data, label in test_iter:
            data = data.reshape(-1, sequence_length, input_size).to(device)
            label = label.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        print('Test Accuracy of the model on the  test : {} %'.format(100 * correct / total))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for index, data, label in train_iter:
            data = data.reshape(-1, sequence_length, input_size).to(device)
            label = label.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        print('Test Accuracy of the model on the  train : {} %'.format(100 * correct / total))
