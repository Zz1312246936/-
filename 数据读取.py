 # gpt写的一个基于PyTorch的LSTM网络，同时使用注意力机制和加权损失函数来处理不均衡的二分类问题
import time
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from My_LSTM import My_LSTM


class myDataSet(Dataset):
    def __init__(self, data_dir, label_dir, ):
        """
        :param data_dir: 数据文件路径
        :param label_dir: 标签文件路径
        :param transform: transform操作
        """

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
        index_data = int(self.data_path[index][13:-4])
        # 读取标签

        # label = label[index, 3]
        label = self.labels.iloc[index_data, 3]


        # 转成张量
        numpy_array = data.values
        data = torch.tensor(numpy_array, dtype=torch.float32)
        # data = torch.tensor(data.values)
        label = torch.tensor(label)

        return index_data, data, label  # 返回数据和标签
class MyDataSet(Dataset):
    def __init__(self, data_dir, label_dir, ):
        """
        :param data_dir: 数据文件路径
        :param label_dir: 标签文件路径
        :param transform: transform操作
        """

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
        index_data = int(self.data_path[index][13:-4])
        # 读取标签

        # label = label[index, 3]
        # label_index = index_data % 7000
        # label = self.labels.iloc[label_index, 3]
        label = self.labels.iloc[index_data, 3]
        # 转成张量
        numpy_array = data.values
        data = torch.tensor(numpy_array, dtype=torch.float32)
        # data = torch.tensor(data.values)
        label = torch.tensor(label)

        return index_data, data, label  # 返回数据和标签
def main_FD():
    data_train_dir = 'F:\\数据集\\train'
    label_train_dir = 'F:\\数据集\\train_label\\metadata_train.csv'
    data_test_dir = 'F:\\数据集\\test1'
    label_test_dir = 'F:\\数据集\\test1_label\\metadata_train.csv'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    sequence_length = 400
    input_size = 34
    hidden_size = 128
    num_layers = 5
    num_classes = 2
    batch_size = 32
    num_epochs = 101
    learning_rate = 0.001
    class_weights = torch.FloatTensor([1, 8]).to(device)
    # 读取数据集
    train_dataset = myDataSet(
        data_dir=data_train_dir,
        label_dir=label_train_dir,
    )
    test_dataset = MyDataSet(
        data_dir=data_test_dir,
        label_dir=label_test_dir,
    )
    train_iter = DataLoader(train_dataset, batch_size)
    val_iter = DataLoader(test_dataset, batch_size)
    model = My_LSTM(input_size, hidden_size, num_layers, num_classes, device).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-4)

    # Train the model
    print("start train")
    best_Precision = 0
    best_MCC = 0
    for epoch in range(num_epochs):
        time_start = time.time()
        lossum = 0
        optimizer.zero_grad()
        for index, data, label in train_iter:
            data = data.reshape(-1, sequence_length, input_size).to(device)
            label = label.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, label)
            lossum += loss


            loss.backward()


        time_epoch = time.time() - time_start
        print('\033[91m Epoch [{}/{}], Lossum: {:.4f}, Time: {:.3f} \033[0m'.format(epoch + 1, num_epochs, lossum.item(), time_epoch))
        optimizer.step()
        scheduler.step(epoch)

        if epoch % 10  == 0 :
            model.eval()
            incorrect_indices = []
            y_true = []
            y_pred = []
            with torch.no_grad():
                for _, data, label in val_iter:
                    data = data.reshape(-1, sequence_length, input_size).to(device)
                    label = label.to(device)
                    # 前向传播
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    # predicted = (outputs > threshold).int()
                    # 比较真实标签和预测标签
                    incorrect_samples = (predicted != label).nonzero()

                    # 收集判断错误的样本索引
                    incorrect_indices.extend(incorrect_samples.squeeze().cpu().numpy().tolist())
                    y_true.extend(label.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
            # 计算准确率和召回率,F1,MCC
            print("Incorrect Sample Indices:", incorrect_indices)
            accuracy = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            confusionmatrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

            print('Validation F1 Score: {:.4f}'.format(f1))
            print('Validation MCC: {:.4f}'.format(mcc))
            print('Validation Accuracy: {:.4f}'.format(accuracy))
            print('Validation Recall: {:.4f}'.format(recall))
            print('Validation Precision: {:.4f}'.format(precision))
            print('Confusion Matrix:')
            print(confusionmatrix)
            if best_MCC < mcc:
                torch.save(model.state_dict(), "res//best_" + 'rnn_model.ckpt')
                best_MCC = mcc
                print('\033[92m 更好！MCC = {:.4f} \033[0m'.format(mcc))
            else:
                print('\033[94m MCC = {:.4f} \033[0m'.format(mcc))
            torch.save(model.state_dict(), "res//{}".format(epoch)+'rnn_model.ckpt')
            # if best_Precision < confusionmatrix[0][0] + confusionmatrix[1][1]:
            #     torch.save(model.state_dict(), "res//best_" + 'rnn_model.ckpt')
            #     best_Precision = confusionmatrix[0][0] + confusionmatrix[1][1]
            #     print('\033[92m Better!!! TN = {}, FN = {}, TP = {}, FP = {} \033[0m'.format(confusionmatrix[0][0], confusionmatrix[1][0], confusionmatrix[1][1], confusionmatrix[0][1]))
            # else:
            #     print('\033[94m TN = {}, FN = {}, TP = {}, FP = {} \033[0m'.format(confusionmatrix[0][0], confusionmatrix[1][0], confusionmatrix[1][1], confusionmatrix[0][1]))
            # torch.save(model.state_dict(), "res//{}".format(epoch)+'rnn_model.ckpt')
            model.train()

    print("start val")
    model.eval()
    y_true = []
    y_pred = []


    with torch.no_grad():
        incorrect_indices = []
        for _, data, label in val_iter:
            data = data.reshape(-1, sequence_length, input_size).to(device)
            label = label.to(device)
            # 前向传播
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            # 比较真实标签和预测标签
            incorrect_samples = (predicted != label).nonzero()

            # 收集判断错误的样本索引
            incorrect_indices.extend(incorrect_samples.squeeze().cpu().numpy().tolist())

            # predicted = (outputs > threshold).int()
            y_true.extend(label.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    # 计算准确率和召回率
    print("Incorrect Sample Indices:", incorrect_indices)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    confusionmatrix = confusion_matrix(y_true, y_pred)

    print('Validation F1 Score: {:.4f}'.format(f1))
    print('Validation MCC: {:.4f}'.format(mcc))
    print('Validation Accuracy: {:.4f}'.format(accuracy))
    print('Validation Recall: {:.4f}'.format(recall))
    print('Validation Precision: {:.4f}'.format(precision))
    print('Confusion Matrix:')
    print(confusionmatrix)
    print('\033[92m  TN = {}, FN = {}, TP = {}, FP = {} \033[0m'.format(confusionmatrix[0][0], confusionmatrix[1][0], confusionmatrix[1][1], confusionmatrix[0][1]))
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for _, data, label in train_iter:
            data = data.reshape(-1, sequence_length, input_size).to(device)
            label = label.to(device)
            # 前向传播
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            # predicted = (outputs > threshold).int()
            y_true.extend(label.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    # 计算准确率和召回率
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    # confusionmatrix = confusion_matrix(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    confusionmatrix = confusion_matrix(y_true, y_pred)

    print('Validation F1 Score: {:.4f}'.format(f1))
    print('Validation MCC: {:.4f}'.format(mcc))
    print('Validation Accuracy: {:.4f}'.format(accuracy))
    print('Validation Recall: {:.4f}'.format(recall))
    print('Validation Precision: {:.4f}'.format(precision))
    print('Confusion Matrix:')
    print(confusionmatrix)
    print('\033[92m  TN = {}, FN = {}, TP = {}, FP = {} \033[0m'.format(confusionmatrix[0][0], confusionmatrix[1][0], confusionmatrix[1][1], confusionmatrix[0][1]))

if __name__ == "__main__":
    main_FD()
    # test_FD(100)
    # train_res(100)

    print("finish")
