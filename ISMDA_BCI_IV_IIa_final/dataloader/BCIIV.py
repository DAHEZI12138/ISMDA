import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
import scipy.io as sio


class Load_Dataset: #load initial dataset
    def __init__(self, Datasets):
        super(Load_Dataset, self).__init__()
        X_train = []
        y_train = []
        for i in Datasets:
            data = MotorImageryDataset(i)
            X_train0, y_train0 = data.get_trials_from_channels()
            X_train = X_train + X_train0
            y_train = y_train + y_train0

        self.num_channels = X_train[0][0].shape[0]
        
        self.len = np.array(X_train, dtype=object).shape[0]


        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
         return self.len


class Load_Dataset_pseudo(Dataset): # load pseudo label in phase 1
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


class Load_Dataset_pseudo_2(Dataset): # load pseudo label in phase 2
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


def data_generator(configs, train_datasetnum):

    # Loading datasets
    train_dataset = ['A01E', 'A01T', 'A02E', 'A02T', 'A03E',  'A03T', 'A04E',  'A04T', 'A05E',  'A05T', 'A06E', 'A06T', 'A07E',  'A07T', 'A08E', 'A08T', 'A09E',  'A09T']
    train_dataset = Load_Dataset(train_dataset[:(train_datasetnum * 2)] + train_dataset[(train_datasetnum * 2 + 2):])
    valid_dataset = ['A01E', 'A01T', 'A02E', 'A02T', 'A03E',  'A03T', 'A04E',  'A04T', 'A05E',  'A05T', 'A06E', 'A06T', 'A07E',  'A07T', 'A08E', 'A08T', 'A09E',  'A09T']
    valid_dataset = Load_Dataset(valid_dataset[(train_datasetnum * 2):(train_datasetnum * 2 + 2)])

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=False, num_workers=0)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=False, num_workers=0)


    return train_loader, valid_loader



class MotorImageryDataset:
    def __init__(self, dataset='A01T.npz'):
        if not dataset.endswith('.npz'):
            dataset += '.npz'
        self.dataset = dataset
        data_path = '../DATA/bcidatasetIV2a-master/'
        dataset1 = os.path.join(data_path, dataset)
        self.data = np.load(dataset1)

        self.Fs = 250  # 250Hz from original paper

        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']
        self.raw = self.data['s'].T
        self.events_type = self.data['etyp'].T
        self.events_position = self.data['epos'].T
        self.events_duration = self.data['edur'].T
        self.artifacts = self.data['artifacts'].T

        # Types of motor imagery
        self.mi_types = {769: 0, 770: 1,
                         771: 2, 772: 3, 783: 'unknown'}

    def get_trials_from_channel(self):

        # Channel default is C3
        if self.dataset.endswith('E.npz'):
            label_path = '../DATA/bcidatasetIV2a-master/true_labels/'
            labels = os.path.join(label_path, self.dataset + '.mat')
            labels = sio.loadmat(labels)["classlabel"]
        startrial_code = 768
        starttrial_events = self.events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]

        trials = []
        classes = []
        k = 0
        for index in idxs:
            try: # mi_types may not have the values generated by type_e
                type_e = self.events_type[0, index+1]
                class_e = self.mi_types[type_e]
                if self.dataset.endswith('T.npz'):
                    if class_e == 'unknown':
                        continue
                else:
                    class_e = (int(labels[k])-1)
                    k = k + 1
                classes.append(class_e)

                start = self.events_position[0, index]
                stop = start + self.events_duration[0, index]
                trial = self.raw[0:22, (start + 500):(stop - 375)]

                trial = trial.reshape((1, 22, 1000))
                trials.append(trial)

            except:
                if self.dataset.endswith('E.npz'):
                    k = k + 1
                continue

        return trials, classes

    def get_trials_from_channels(self):
        t, classes = self.get_trials_from_channel()

        return t, classes # Returns all trials with corresponding tags


