import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
import h5py

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



def data_generator_54(configs, train_datasetnum):

    # Loading datasets
    train_dataset = Load_Dataset(train_datasetnum, True)
    valid_dataset = Load_Dataset(train_datasetnum, False)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=False, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=False, num_workers=0)
    return train_loader, valid_loader







class Load_Dataset:
    # Initialize your data, download, etc.
    def __init__(self, subject_num, Train):
        super(Load_Dataset, self).__init__()
        subjs = [1, 47, 46, 37, 13, 27, 12, 32, 53, 54, 4, 40, 19, 41, 18, 42, 34, 7,
                 49, 9, 5, 48, 29, 15, 21, 17, 31, 45, 35, 38, 51, 8, 11, 16, 28, 44, 24,
                 52, 3, 26, 39, 50, 6, 23, 2, 14, 25, 20, 10, 33, 22, 43, 36, 30]
        if Train is True:
            cv_set = np.array(subjs[subject_num + 1:] + subjs[:subject_num])

            train_subjs = cv_set
            X_train, Y_train = get_multi_data(train_subjs)

        else:
            valid_subjs = subjs[subject_num]
            X_val, Y_val = get_data(valid_subjs)
            X_train, Y_train = X_val[300:], Y_val[300:]

        X_train = X_train.reshape(-1, 1, 62, 1000)


        self.num_channels = X_train.shape[2]

        self.len = X_train.shape[0]

        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # 返回人数
        return self.len



def get_data(subj):
    filepath_session = "../DATA/new/KU_mi_smt.h5"
    dfile = h5py.File(filepath_session, 'r')
    dpath = '/s' + str(subj)
    X = dfile[os.path.join(dpath + '/X')]
    Y = dfile[os.path.join(dpath + '/Y')]
    return X[:], Y[:]

def get_multi_data(subjs):
    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data(s)
        Xs.append(x[:])
        Ys.append(y[:])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y