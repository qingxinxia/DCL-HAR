import os.path

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import seaborn as sns
import matplotlib.pyplot as plt


# def sliding_window(arr, window_size, step):
#     # Number of windows that can be created
#     n_windows = (len(arr) - window_size) // step + 1
#     # Creating an empty array to store the windows
#     if isinstance(arr, np.ndarray):
#         output = np.empty((n_windows, window_size, arr.shape[1]))
#     elif isinstance(arr, list):
#         output = np.empty((n_windows, window_size, 1))
#     else:
#         print('check arr type please!')
#         return
#     for i in range(n_windows):
#         start = i * step
#         end = start + window_size
#         output[i] = arr[start:end]
#     return output

# Function to generate new array using sliding window over the first dimension
def sliding_window(arr, window_size, step, islabel=False):
    # Number of windows that can be created
    n_windows = (len(arr) - window_size) // step + 1
    # Creating an empty array to store the windows
    if not islabel:
    # if isinstance(arr, np.ndarray):
        output = np.empty((n_windows, window_size, arr.shape[1]))
    else:
    # elif isinstance(arr, list):
        output = list(np.empty((n_windows, window_size, 1)))
    # else:
    #     print('check arr type please!')
    #     return
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

def dict_mmfit():
    mmfit_sentences = {
    1: 'bicep_curls',
    2: 'dumbbell_rows',
    3: 'dumbbell_shoulder_press',
    4: 'jumping_jacks',
    5: 'lateral_shoulder_raises',
    6: 'lunges',
    7: 'pushups',
    8: 'situps',
    9: 'squats',
    10: 'tricep_extensions',
    0: 'no_action'
    }
    return mmfit_sentences

def get_virtual_imu_openpack(test_dataset):
    openpack_sentences = dict_openpack()

    data_root_path = os.path.join(os.getcwd(), 'data', 'raw', 'openpack')
    all_datalist = ['U0201', 'U0206', 'U0207', 'U0208', 'U0209']
    imu_train, label_train = [], []
    imu_test, label_test = [], []
    for u in all_datalist:
        data_folder = os.path.join(data_root_path, 'CNNMLP_seg_openpack_testset_%s_epoch_99'%u)
        with open(os.path.join(data_folder, 'pred_acc_seg.npy'), 'rb') as f:
            acc = np.load(f)
            # print(acc.shape)
        with open(os.path.join(data_folder, 'pred_gyro_seg.npy'), 'rb') as f:
            gyr = np.load(f)
        data = np.concatenate((acc, gyr), axis=2)
        # print(data.shape)
        reshapedata = data.reshape(-1, data.shape[-1])

        with open(os.path.join(data_folder, 'pred_label_seg.npy'), 'rb') as f:
            label = np.load(f, allow_pickle=True)
            nan_indices = pd.isna(label)
            label[nan_indices] = 0  # Replace NaNs with zero
            for key, value in openpack_sentences.items():
                l_indices = np.where(label == value)[0]
                label[l_indices] = key

        if u != test_dataset:
            imu_train.append(reshapedata)
            label_train.append(label)
        else:
            imu_test.append(reshapedata)
            label_test.append(label)
    return imu_train, label_train, imu_test, label_test

def get_virtual_imu_mmfit():
    mmfit_sentences = dict_mmfit()

    imu_path = os.path.join(os.getcwd(), 'data', 'raw', 'mmfit')
    data_folder = os.path.join(imu_path, 'CNNMLP_seg_mmfit_epoch_100')
    with open(os.path.join(data_folder, 'pred_acc_seg.npy'), 'rb') as f:
        acc = np.load(f)
    with open(os.path.join(data_folder, 'pred_gyro_seg.npy'), 'rb') as f:
        gyr = np.load(f)
    data = np.concatenate((acc, gyr), axis=2)
    reshapedata = data.reshape(-1, data.shape[-1])

    with open(os.path.join(data_folder, 'GTlabel_seg.npy'), 'rb') as f:
        label = np.load(f, allow_pickle=True)
        nan_indices = pd.isna(label)
        label[nan_indices] = 0  # Replace NaNs with zero
        for key, value in mmfit_sentences.items():
            l_indices = np.where(label == value)[0]
            label[l_indices] = key

    return reshapedata, label.reshape(-1)

def prepare_data_mmfit(batch_size, datalen, train_dataset, both_wrists, virtual_IMU, device):
    mmfit_sentences = dict_mmfit()

    TRAIN_W_IDs = ['01', '02', '03', '04', '06', '07', '08', '16', '17', '18']
    # VAL_W_IDs = ['14', '15', '19']
    TEST_W_IDs = ['00', '05', '12', '13', '20']

    imu_path = os.path.join(os.getcwd(), 'data', 'raw', 'mmfit', 'imu_real')

    # train loader
    imu_train, joint_train, label_train = [], [], []
    for name in TRAIN_W_IDs:
        # 当train dataset空，都用
        if (name in train_dataset) or (len(train_dataset) == 0):
        # if name in train_dataset:
            imup = os.path.join(imu_path, 'w%s.np' % name)
            tmp = np.load(imup, allow_pickle=True)
            if not both_wrists:
                imu_data = tmp[:, 1:4]
            else:
                imu_data = tmp[:, 1:13]
            label_data = tmp[:, -1]
            imu_train.append(imu_data)
            label_str = np.array([str(i) for i in label_data])
            for key, value in mmfit_sentences.items():
                l_indices = np.where(label_str == value)[0]
                label_str[l_indices] = key
            label_train.extend(label_str)

    # # validataion loader
    # imu_val, joint_val, label_val = [], [], []
    # for name in VAL_W_IDs:
    #     imup = os.path.join(imu_path, 'w%s.np' % name)
    #     tmp = np.load(imup, allow_pickle=True)
    #     if not both_wrists:
    #         imu_data = tmp[:, 1:4]
    #     else:
    #         imu_data = tmp[:, 1:13]
    #     label_data = tmp[:, -1]
    #     imu_val.append(imu_data)
    #     label_str = np.array([str(i) for i in label_data])
    #     for key, value in mmfit_sentences.items():
    #         l_indices = np.where(label_str == value)[0]
    #         label_str[l_indices] = key
    #     label_val.extend(label_str)

    # test loader
    imu_test, joint_test, label_test = [], [], []
    for name in TEST_W_IDs:
        imup = os.path.join(imu_path, 'w%s.np' % name)
        tmp = np.load(imup, allow_pickle=True)
        if not both_wrists:
            imu_data = tmp[:, 1:4]
        else:
            imu_data = tmp[:, 1:13]
        label_data = tmp[:, -1]
        imu_test.append(imu_data)
        label_str = np.array([str(i) for i in label_data])
        for key, value in mmfit_sentences.items():
            l_indices = np.where(label_str == value)[0]
            label_str[l_indices] = key
        label_test.extend(label_str)

    if virtual_IMU:  # for training
        vimu_train, vlabel_train = \
            get_virtual_imu_mmfit()
        if not both_wrists:
            imu_train.append(vimu_train[:, 1:4])
        else:
            imu_train.append(vimu_train)
        label_train.extend(vlabel_train)


    # prepare dataloader
    whole_imu_train1 = np.concatenate(imu_train)
    ## todo: downsampling from 100Hz to 30Hz
    # Calculate the downsample factor
    downsample_factor = int(100 / 30)  # This should be 3 for 100Hz to 30Hz
    # Select every 3rd sample along the first axis (time axis)
    whole_imu_train = whole_imu_train1[::downsample_factor, :]
    label_train = label_train[::downsample_factor]


    imu_data_win = sliding_window(whole_imu_train, datalen, int(datalen / 2))
    label_data_win = sliding_window(label_train, datalen, int(datalen / 2), islabel=True)
    train_set = base_dataset(imu_data_win, label_data_win, device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=0)

    # whole_imu_val = np.concatenate(imu_val)
    # imu_data_win = sliding_window(whole_imu_val, datalen, int(datalen / 2))
    # label_data_win = sliding_window(label_str, datalen, int(datalen / 2), islabel=True)
    # val_set = base_dataset(imu_data_win,
    #                         label_data_win)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,
    #                         pin_memory=True, num_workers=1)

    whole_imu_test1 = np.concatenate(imu_test)
    ## todo: downsampling from 100Hz to 30Hz
    # Select every 3rd sample along the first axis (time axis)
    whole_imu_test = whole_imu_test1[::downsample_factor, :]
    label_test = label_test[::downsample_factor]

    imu_data_win = sliding_window(whole_imu_test, datalen, int(datalen / 2))
    label_data_win = sliding_window(label_test, datalen, int(datalen / 2), islabel=True)
    test_set = base_dataset(imu_data_win, label_data_win, device)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    return train_loader, [], test_loader

def prepare_data_mmfit_imutube(batch_size, datalen, train_dataset, both_wrists, virtual_IMU, device):
    mmfit_sentences = dict_mmfit()

    TRAIN_W_IDs = ['01', '02', '03', '04', '06', '07', '08', '16', '17', '18']
    # VAL_W_IDs = ['14', '15', '19']
    TEST_W_IDs = ['00', '05', '12', '13', '20']

    imu_path = os.path.join(os.getcwd(), 'data', 'raw', 'mmfit', 'imu_real')

    # train loader
    imu_train, joint_train, label_train = [], [], []
    for name in TRAIN_W_IDs:
        # 当train dataset空，都用
        if (name in train_dataset) or (len(train_dataset) == 0):
        # if name in train_dataset:
            imup = os.path.join(imu_path, 'w%s.np' % name)
            tmp = np.load(imup, allow_pickle=True)
            if not both_wrists:
                imu_data = tmp[:, 1:4]
            else:
                imu_data = tmp[:, [1,2,3,7,8,9]]
            label_data = tmp[:, -1]
            imu_train.append(imu_data)
            label_str = np.array([str(i) for i in label_data])
            for key, value in mmfit_sentences.items():
                l_indices = np.where(label_str == value)[0]
                label_str[l_indices] = key
            label_train.extend(label_str)

    # test loader
    imu_test, joint_test, label_test = [], [], []
    for name in TEST_W_IDs:
        imup = os.path.join(imu_path, 'w%s.np' % name)
        tmp = np.load(imup, allow_pickle=True)
        if not both_wrists:
            imu_data = tmp[:, 1:4]
        else:
            imu_data = tmp[:, [1,2,3,7,8,9]]
        label_data = tmp[:, -1]
        imu_test.append(imu_data)
        label_str = np.array([str(i) for i in label_data])
        for key, value in mmfit_sentences.items():
            l_indices = np.where(label_str == value)[0]
            label_str[l_indices] = key
        label_test.extend(label_str)

    if virtual_IMU:  # for training
        imu_path = os.path.join(os.getcwd(), 'data', 'raw', 'mmfit', 'imuTubeData.npz')
        data = np.load(imu_path)
        if not both_wrists:
            imu_train.append(data['x'])
            label_train.extend(data['y'].astype(int))
        else:
            imu_pathr = os.path.join(os.getcwd(), 'data', 'raw', 'mmfit', 'imuTubeDataRight.npz')
            datar = np.load(imu_pathr)
            imu_train.append(np.concatenate([data['x'], datar['x']], axis=1))
            label_train.extend(data['y'].astype(int))

    # prepare dataloader
    whole_imu_train1 = np.concatenate(imu_train)
    ## todo: downsampling from 100Hz to 30Hz
    # Calculate the downsample factor
    downsample_factor = int(100 / 30)  # This should be 3 for 100Hz to 30Hz
    # Select every 3rd sample along the first axis (time axis)
    whole_imu_train = whole_imu_train1[::downsample_factor, :]
    label_train = label_train[::downsample_factor]


    imu_data_win = sliding_window(whole_imu_train, datalen, int(datalen / 2))
    label_data_win = sliding_window(label_train, datalen, int(datalen / 2), islabel=True)
    train_set = base_dataset(imu_data_win, label_data_win, device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=0)


    whole_imu_test1 = np.concatenate(imu_test)
    ## todo: downsampling from 100Hz to 30Hz
    # Select every 3rd sample along the first axis (time axis)
    whole_imu_test = whole_imu_test1[::downsample_factor, :]
    label_test = label_test[::downsample_factor]

    imu_data_win = sliding_window(whole_imu_test, datalen, int(datalen / 2))
    label_data_win = sliding_window(label_test, datalen, int(datalen / 2), islabel=True)
    test_set = base_dataset(imu_data_win, label_data_win, device)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    return train_loader, [], test_loader

def prepare_data_openpack(dataset, batch_size, datalen, train_dataset,
                          test_dataset, both_wrists, virtual_IMU, device):
    # load data as numpy type, data shape=(batch, datalen, datadim) with overlapping
    # output data,label
    # only S0100 dat now.....
    openpack_sentences = dict_openpack()

    data_root_path = os.path.join(os.getcwd(), 'data', 'raw', dataset)
    # real IMU data as
    groundtruth_data_paths_train, groundtruth_data_paths_test = [], []
    for filename in os.listdir(data_root_path):
        if filename.endswith('.npy'):
            u = filename.split("IMU_")[1].split("_S")[0]
            if test_dataset in filename:
                groundtruth_data_paths_test.append(os.path.join(data_root_path, filename))
            elif u in train_dataset:
                groundtruth_data_paths_train.append(os.path.join(data_root_path, filename))

    # read data
    imu_train, label_train = [], []
    for imu in groundtruth_data_paths_train:
        with open(imu, 'rb') as f:
            tmp_data = np.load(f, allow_pickle=True)
            # print('dfsf'+str(type(both_wrists)))
            if both_wrists:
                # print('a')
                imu_data = tmp_data[:, 1:13].reshape(tmp_data.shape[0], -1)
            else:
                # print('b')
                imu_data = tmp_data[:, 1:4].reshape(tmp_data.shape[0], -1)
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
            # print('ddddd' + str(type(both_wrists)))
            if both_wrists==True:
                # print('test both')
                imu_data = tmp_data[:, 1:13].reshape(tmp_data.shape[0], -1)
            else:
                # print('test single')
                imu_data = tmp_data[:, 1:4].reshape(tmp_data.shape[0], -1)
            imu_label = tmp_data[:, -1]
        # concatenate users
        imu_test.append(imu_data)
        nan_indices = pd.isna(imu_label)
        imu_label[nan_indices] = 0  # Replace NaNs with zero
        for key, value in openpack_sentences.items():
            l_indices = np.where(imu_label == value)[0]
            imu_label[l_indices] = key
        label_test.append(imu_label)

    # if add virtual data
    if virtual_IMU:
        vimu_train, vlabel_train, vimu_test, vlabel_test = \
            get_virtual_imu_openpack(test_dataset)
        # print(vimu_train[0].shape)
        if both_wrists:
            # print(both_wrists)
            vd = [v for v in vimu_train]
            vt = [v for v in vimu_test]
            # print(vd[0].shape)
            # print(vt[0].shape)
        else:
            vd = [v[:, 1:4] for v in vimu_train]
            vt = [v[:, 1:4] for v in vimu_test]

        imu_train.extend(vd)
        label_train.extend(vlabel_train)
        imu_test.extend(vt)
        label_test.extend(vlabel_test)

    whole_imu_train = np.concatenate(imu_train)
    whole_label_train = np.concatenate(label_train)
    whole_imu_train_win = sliding_window(whole_imu_train, datalen, int(datalen/2))
    whole_label_train_win = sliding_window(whole_label_train.reshape(-1, 1),
                                           datalen, int(datalen/2))
    train_set = base_dataset(whole_imu_train_win, whole_label_train_win, device)

    whole_imu_test = np.concatenate(imu_test)
    whole_label_test = np.concatenate(label_test)
    whole_imu_test_win = sliding_window(whole_imu_test, datalen, datalen)
    whole_label_test_win = sliding_window(whole_label_test.reshape(-1, 1),
                                          datalen, datalen)

    test_set = base_dataset(whole_imu_test_win, whole_label_test_win, device)
    # Create a DataLoader, drop_last=True to avoid bug
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=0)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)

    return train_loader, test_loader


class base_dataset(Dataset):
    def __init__(self, imu, label, device='cuda:0'):
        self.imu = torch.from_numpy(imu.astype(float))  # activity label of the sensor segment
        self.label = torch.tensor(np.array(label).astype(int))  # filename of the data belongs to
        # self.label = torch.tensor(label)  # filename of the data belongs to
        self.imu = self.imu.to(device=device, non_blocking=True, dtype=torch.float)
        self.label = self.label.to(device=device, non_blocking=True, dtype=torch.int)

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