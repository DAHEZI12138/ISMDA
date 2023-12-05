import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import scipy.io as sio

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

def data_generator_HG(configs, train_datasetnum):

    train_dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    train_dataset = Load_Dataset_HG(train_dataset[:train_datasetnum] + train_dataset[(train_datasetnum + 1):])
    valid_dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    valid_dataset = Load_Dataset_HG(valid_dataset[train_datasetnum:(train_datasetnum + 1)])

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=False, num_workers=0, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    return train_loader, valid_loader

class Load_Dataset_HG:
    # Initialize your data, download, etc.
    def __init__(self, Datasets):
        super(Load_Dataset_HG, self).__init__()
        X_train = []
        y_train = []
        filepath = "../DATA/HGD/mat"
        for i in Datasets:
            filepath0 = filepath + '/' + str(i) + '_T'
            filepath1 = filepath + '/' + str(i) + '_E'
            X_train0 = sio.loadmat(filepath0)['data']
            y_train0 = sio.loadmat(filepath0)['label']
            X_train1 = sio.loadmat(filepath1)['data']
            y_train1 = sio.loadmat(filepath1)['label']
            X_train0 = X_train0.reshape((-1, 1, 44, 1000))
            X_train1 = X_train1.reshape((-1, 1, 44, 1000))
            for j in range(X_train0.shape[0]):
                X_train.append(X_train0[j, :, :, :])
                y_train.append(y_train0[0, j])
            for k in range(X_train1.shape[0]):
                X_train.append(X_train1[k, :, :, :])
                y_train.append(y_train1[0, k])


        self.num_channels = X_train[0][0].shape[0]

        self.len = np.array(X_train, dtype=object).shape[0]

        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Load_Dataset_pseudo(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(Load_Dataset_pseudo, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]
        Z_train = dataset["softmax_labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        self.num_channels = min(X_train.shape)
        if X_train.shape.index(self.num_channels) != 1:  # data dim is #samples, seq_len, #channels
            X_train = X_train.permute(0, 2, 1)

        self.len = X_train.shape[0]

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
            self.z_data = torch.from_numpy(Z_train)
        else:
            self.x_data = X_train.float()
            self.y_data = y_train
            self.z_data = Z_train.float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], index, self.z_data[index]

    def __len__(self):
        return self.len


class Load_Dataset_pseudo_2(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(Load_Dataset_pseudo_2, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]
        Z_train = dataset["softmax_labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        self.num_channels = min(X_train.shape)
        if X_train.shape.index(self.num_channels) != 1:
            X_train = X_train.permute(0, 2, 1)

        self.len = X_train.shape[0]

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
            self.z_data = torch.from_numpy(Z_train)
        else:
            self.x_data = X_train.float()
            self.y_data = y_train
            self.z_data = Z_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], index, self.z_data[index]

    def __len__(self):
        return self.len