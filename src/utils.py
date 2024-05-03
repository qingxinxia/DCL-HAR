import os.path

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import seaborn as sns
import matplotlib.pyplot as plt


def sliding_window(arr, window_size, step):
    # Number of windows that can be created
    n_windows = (len(arr) - window_size) // step + 1
    # Creating an empty array to store the windows
    if isinstance(arr, np.ndarray):
        output = np.empty((n_windows, window_size, arr.shape[1]))
    elif isinstance(arr, list):
        output = np.empty((n_windows, window_size, 1))
    else:
        print('check arr type please!')
        return
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        output[i] = arr[start:end]
    return output

def dict_openpack():
    # define None class label = 0
    openpack_sentences = {
        1: 'Picking',
        2: 'Relocate Item Label',
        3: 'Assemble Box',
        4: 'Insert Items',
        5: 'Close Box',
        6: 'Attach Box Label',
        7: 'Scan Label',
        8: 'Attach Shipping Label',
        9: 'Put on Back Table',
        10: 'Fill out Order',
        0: 'Null'
    }
    return openpack_sentences

def prepare_data_openpack(dataset, batch_size, datalen, test_dataset):
    # load data as numpy type, data shape=(batch, datalen, datadim) with overlapping
    # output data,label
    # only S0100 dat now.....
    openpack_sentences = dict_openpack()

    data_root_path = os.path.join(os.getcwd(), 'data', 'raw', dataset)
    # real IMU data as
    groundtruth_data_paths_train, groundtruth_data_paths_test = [], []
    for filename in os.listdir(data_root_path):
        if filename.endswith('.npy'):
            if test_dataset in filename:
                groundtruth_data_paths_test.append(os.path.join(data_root_path, filename))
            else:
                groundtruth_data_paths_train.append(os.path.join(data_root_path, filename))

    # read data
    imu_train, label_train = [], []
    for imu in groundtruth_data_paths_train:
        with open(imu, 'rb') as f:
            tmp_data = np.load(f, allow_pickle=True)
            imu_data = tmp_data[:, 1:13].reshape(tmp_data.shape[0], -1)
            imu_label = tmp_data[:, -1]
        # concatenate users
        imu_train.append(imu_data)
        nan_indices = pd.isna(imu_label)
        imu_label[nan_indices] = 0  # Replace NaNs with zero
        for key, value in openpack_sentences.items():
            l_indices = np.where(imu_label == value)[0]
            imu_label[l_indices] = key
        label_train.append(imu_label)

    imu_test, label_test = [], []
    for imu in groundtruth_data_paths_test:
        with open(imu, 'rb') as f:
            tmp_data = np.load(f, allow_pickle=True)
            imu_data = tmp_data[:, 1:13].reshape(tmp_data.shape[0], -1)
            imu_label = tmp_data[:, -1]
        # concatenate users
        imu_test.append(imu_data)
        nan_indices = pd.isna(imu_label)
        imu_label[nan_indices] = 0  # Replace NaNs with zero
        for key, value in openpack_sentences.items():
            l_indices = np.where(imu_label == value)[0]
            imu_label[l_indices] = key
        label_test.append(imu_label)

    whole_imu_train = np.concatenate(imu_train)
    whole_label_train = np.concatenate(label_train)
    whole_imu_train_win = sliding_window(whole_imu_train, datalen, int(datalen/2))
    whole_label_train_win = sliding_window(whole_label_train.reshape(-1,1), datalen, int(datalen/2))
    train_set = base_dataset(whole_imu_train_win, whole_label_train_win)

    whole_imu_test = np.concatenate(imu_test)
    whole_label_test = np.concatenate(label_test)
    whole_imu_test_win = sliding_window(whole_imu_test, datalen, datalen)
    whole_label_test_win = sliding_window(whole_label_test.reshape(-1,1), datalen, datalen)

    test_set = base_dataset(whole_imu_test_win, whole_label_test_win)
    # Create a DataLoader, drop_last=True to avoid bug
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)

    return train_loader, test_loader


class base_dataset(Dataset):
    def __init__(self, imu, label):
        self.imu = torch.from_numpy(imu.astype(float))  # activity label of the sensor segment
        self.label = torch.tensor(label)  # filename of the data belongs to

    def __getitem__(self, index):
        label, imu = self.label[index], self.imu[index]
        return imu, label

    def __len__(self):
        return len(self.label)

def tsne(latent, y_ground_truth, save_dir):
    """
        Plot t-SNE embeddings of the features
    """
    # latent = latent.cpu().detach().numpy()
    # y_ground_truth = y_ground_truth.cpu().detach().numpy()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(latent)
    plt.figure(figsize=(16,10))
    set_y = set(y_ground_truth)
    num_labels = len(set_y)
    sns_plot = sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=y_ground_truth,
        palette=sns.color_palette("hls", num_labels),
        legend="full",
        alpha = 0.5
        )

    sns_plot.get_figure().savefig(save_dir)


def mds(latent, y_ground_truth, save_dir):
    """
        Plot MDS embeddings of the features
    """
    # latent = latent.cpu().detach().numpy()
    mds = MDS(n_components=2)
    mds_results = mds.fit_transform(latent)
    plt.figure(figsize=(16,10))
    set_y = set(y_ground_truth)
    num_labels = len(set_y)
    sns_plot = sns.scatterplot(
        x=mds_results[:,0], y=mds_results[:,1],
        hue=y_ground_truth,
        palette=sns.color_palette("hls", num_labels),
        # data=df_subset,
        legend="full",
        alpha=0.5
        )

    sns_plot.get_figure().savefig(save_dir)