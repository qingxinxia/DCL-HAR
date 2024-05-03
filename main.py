import argparse
import torch
import os
import torch.nn as nn
import numpy as np

from src.utils import prepare_data_openpack
from models.dcl import HARModel, DeepConvLSTM
from src.train_utils import train_time_series_seg, eval_time_series_seg, test

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-dataset', default='openpack', type=str,
                    help='path to dataset')
parser.add_argument('--test_dataset', default='U0201', type=str,
                    help='The test user used in Leave-one-user-out experiment. ')
parser.add_argument('--datalen', default=60, type=int,
                    help='Length of input data(15Hz). ')

parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--in_feature', default=12, type=int,
                    help='number of values (dimensions) of input (12=arms.dim+hands.dim). ')
parser.add_argument('--num_feature', default=128, type=int,
                    help='number of units for network layers')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
# parser.add_argument('--seed', default=None, type=int,
#                     help='seed for initializing training. ')



NB_CLASSES = 11


def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    folder_path = os.path.join(os.getcwd(),
                             'prediction_result',
                             'DCL_%s_test_%s_epoch_%s' % (args.dataset,
                                             args.test_dataset,
                                             str(args.epochs)))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # get train and test dataset/dataloader
    # data.shape=(batch, data length, data dimension)
    train_dataloader, test_dataloader = prepare_data_openpack(args.dataset,
                                                              args.batch_size,
                                                              args.datalen,
                                                              args.test_dataset)

    # model = HARModel(args.in_feature, n_hidden=args.num_feature, n_layers=1,
    #                  n_filters=64, n_classes=NB_CLASSES, filter_size=5,
    #                  datalen=args.datalen, drop_prob=0.5, device=device)

    model = DeepConvLSTM(n_channels=args.in_feature, n_classes=NB_CLASSES,
                         conv_kernels=64, kernel_size=5,
                         LSTM_units=128, backbone=False)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    loss_list = []
    for epoch in range(start_epoch, args.epochs):
        print('Epoch %d/%d' % (epoch, args.epochs))

        loss_value = train_time_series_seg(model, optimizer, criterion,
                                           train_dataloader, device,
                                           args.batch_size)
        loss_list.append(loss_value)
        print("Epoch: {}/{}...".format(epoch + 1, args.epochs),
              "Train Loss: {:.4f}...".format(loss_value))

        if epoch % 100 == 0:
            test(test_dataloader, model, device, criterion,
                 NB_CLASSES, folder_path, epoch, plt=True, savef=False)

    # Save final model

    torch.save(model.state_dict(), 'DCL_%s_test_%s_epoch_%s.pt' % (args.dataset,
                                             args.test_dataset,
                                             str(args.epochs)))

    # check if the latent representation of time series segment is similar when the sensor data is similar
    # GTimu_list, GTlabel, predlabel \
    #     = eval_time_series_seg(model, criterion, test_dataloader, device, args.batch_size)

    GTimu_list, GTlabel, predlabel \
        = test(test_dataloader, model, device, criterion,
               NB_CLASSES, folder_path, epoch, plt=True, savef=True)

    GTimus = np.concatenate(GTimu_list, axis=0)
    GTlabels = np.concatenate(GTlabel, axis=0)
    predlabels = np.concatenate(predlabel, axis=0)

    print('Saving valuables ...')

    # Construct the full path to the file
    file_path = os.path.join(folder_path, 'predlabels_seg.npy')
    with open(file_path, 'wb') as f:
        np.save(f, predlabels)
    file_path = os.path.join(folder_path, 'GTlabels_seg.npy')
    with open(file_path, 'wb') as f:
        np.save(f, GTlabels)
    file_path = os.path.join(folder_path, 'GTimus_seg.npy')
    with open(file_path, 'wb') as f:
        np.save(f, GTimus)  # [left acc3, gyro3, right acc3, gyro3] = 12
    file_path = os.path.join(folder_path, 'loss_seg.npy')
    with open(file_path, 'wb') as f:
        np.save(f, loss_list)


if __name__ == "__main__":
    main()
