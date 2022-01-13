import numpy as np
import torch
from torch.utils.data.dataset import Dataset


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


def normalization(data, normalize=2):
    n, m = data.shape
    scales = np.ones(m, dtype=float)

    if (normalize == 0):
        data = data

    if (normalize == 1):
        data = data / np.max(data)

    # normlized by the maximum value of each row(sensor).
    if (normalize == 2):
        for i in range(m):
            scales[i] = np.max(np.abs(data[:, i]))
            data[:, i] = data[:, i] / scales[i]

    return data, scales


def batchify(data, idx_set, window, horizon):
    m = data.shape[1]
    n = len(idx_set)
    x = np.zeros((n, window, m))
    y = np.zeros((n, m))
    for i in range(n):
        end = idx_set[i] - horizon + 1
        start = end - window
        x[i, :, :] = data[start:end, :]
        y[i, :] = data[idx_set[i], :]
    return [x, y]


def split(data, train_portion, valid_portion, window, horizon):
    n, m = data.shape
    train_int, valid_int = int(train_portion * n), int((valid_portion + train_portion) * n)
    train_set = range(window + horizon - 1, train_int)
    valid_set = range(train_int, valid_int)
    test_set = range(valid_int, n)

    train_data = batchify(data, train_set, window, horizon)
    valid_data = batchify(data, valid_set, window, horizon)
    test_data = batchify(data, test_set, window, horizon)
    return train_data, valid_data, test_data


def Data(filename, normalize=2, train_portion=0.6, valid_portion=0.2, window=3, horizon=3):
    # Step1 : load data
    data = np.loadtxt(filename, delimiter=",")[:, :]
    # Step2 : normalization
    data, scales = normalization(data, normalize=normalize)
    # Step3 : train/valid/test split
    train_data, valid_data, test_data = split(data, train_portion, valid_portion, window, horizon)

    tmp = test_data[1] * np.tile(scales, (test_data[1].shape[0], 1))
    rse_test = normal_std(tmp)
    rae_test = np.mean(abs(tmp - np.mean(tmp)))

    tmp = valid_data[1] * np.tile(scales, (valid_data[1].shape[0], 1))
    rse_val = normal_std(tmp)
    rae_val = np.mean(abs(tmp - np.mean(tmp)))

    return train_data, valid_data, test_data, scales, rse_val, rae_val, rse_test, rae_test


class SingleStepDataset(Dataset):

    def __init__(self, data):
        # self.X : num_samples, window, feature_dim, np.array
        # self.y : num_samples, feature_dim, np.array
        self.X, self.y = data
        self.length = self.X.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.float), \
               torch.tensor(self.y[i], dtype=torch.float)
