import argparse
import torch
import os
import torch.nn as nn
import numpy as np

from src.utils import prepare_data_openpack
from models.dcl import DeepConvLSTMSelfAttn, DeepConvLstmV3
from src.train_utils import train_time_series_seg, eval_time_series_seg, test

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--model', default='DeepConvLstmV3', type=str,
                    help='DeepConvLstmV3, DeepConvLSTMSelfAttn')
parser.add_argument('--dataset', default='openpack', type=str,
                    help='path to dataset')
parser.add_argument('--train_dataset', default=['U0206', 'U0207', 'U0208', 'U0209'], type=list,
                    help='The test user used in Leave-one-user-out experiment. ')
parser.add_argument('--test_dataset', default="U0201", type=str,
                    help='The test user used in Leave-one-user-out experiment. ')
parser.add_argument('--datalen', default=60, type=int,
                    help='Length of input data(15Hz). len60 ')

parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--num_feature', default=128, type=int,
                    help='number of units for network layers')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--checkpoint', default=False, type=bool,
                    help='If load checkpoint or not. ')
parser.add_argument('--virtual_IMU', default=True, type=bool,
                    help='If load virtual IMU or not. ')
# parser.add_argument('--seed', default=None, type=int,
#                     help='seed for initializing training. ')
parser.add_argument('--plts', default=True, type=bool,
                    help='If plot results or not. ')
parser.add_argument('--device', default='cuda:0', type=str,
                    help='If plot results or not. ')
parser.add_argument('--both_wrists', default=False, type=bool,
                    help='If plot results or not. ')
parser.add_argument('--in_feature', default=3, type=int,
                    help='number of values (dimensions) of input (12=arms.dim+hands.dim). ')

NB_CLASSES = 11


def main():
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)
    # print('training set: ')
    # print(args.train_dataset)
    # print('test set: ')
    # print(args.test_dataset)
    # print('use_virtual: %s' % args.virtual_IMU)
    folder_path = os.path.join(os.getcwd(),
                               'prediction_result',
                               'DCL_realvirtual_%s_test_%s_epoch_%s' % (args.dataset,
                                                                        args.test_dataset,
                                                                        str(args.epochs)))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # get train and test dataset/dataloader
    # data.shape=(batch, data length, data dimension)

    if args.model == 'DeepConvLSTMSelfAttn':
        model = DeepConvLSTMSelfAttn(in_ch=args.in_feature, num_classes=NB_CLASSES)
    elif args.model == 'DeepConvLstmV3':
        model = DeepConvLstmV3(in_ch=args.in_feature, num_classes=NB_CLASSES)

    # # load checkpoint
    # if args.checkpoint:
    #     model.load_state_dict(torch.load('DCL_%s_test_%s_epoch_3000.pth' % (args.dataset,
    #                                                                         args.test_dataset)))

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    f1_scores = []
    all_datasets = ['U0201', 'U0206', 'U0207', 'U0208', 'U0209']
    train_list = ['U0201', 'U0206', 'U0207']
    test_dataset = ''
    for i in range(len(all_datasets)):
        train_list[0] = all_datasets[i]
        for j in range(i + 1, len(train_list), 1):
            train_list[1] = all_datasets[j]
            for k in range(j + 1, len(train_list), 1):
                train_list[2] = all_datasets[k]
                for l in range(len(all_datasets)):
                    if (l != i) and (l != j) and (l != k):
                        test_dataset = all_datasets[k]

                        train_dataloader, test_dataloader = prepare_data_openpack(args.dataset,
                                                                                  args.batch_size,
                                                                                  args.datalen,
                                                                                  train_list,
                                                                                  test_dataset,
                                                                                  args.both_wrists,
                                                                                  args.virtual_IMU)

                        start_epoch = 0
                        loss_list = []
                        for epoch in range(start_epoch, args.epochs):
                            # print('Epoch %d/%d' % (epoch, args.epochs))

                            loss_value, accuracy = train_time_series_seg(model, optimizer, criterion,
                                                                         train_dataloader, device,
                                                                         args.batch_size)
                            loss_list.append(loss_value)


                        # # Save final model
                        # torch.save(model.state_dict(), 'DCL20_realvirtual_both_%s_test_%s_epoch_%s.pth' % (args.dataset,
                        #                                                                             args.test_dataset,
                        #                                                                             str(args.epochs)))

                        # check if the latent representation of time series segment is similar when the sensor data is similar
                        GTimu_list, GTlabel, predlabel, acc_test \
                            = test(test_dataloader, model, device, criterion,
                                   NB_CLASSES, folder_path, epoch, plts=args.plts, savef=True)
                        f1_scores.append(acc_test)
                        print(acc_test)
                        # GTimus = np.concatenate(GTimu_list, axis=0)
                        # GTlabels = np.concatenate(GTlabel, axis=0)
                        # predlabels = np.concatenate(predlabel, axis=0)
                        #
                        # print('Saving valuables ...')
                        #
                        # # Construct the full path to the file
                        # file_path = os.path.join(folder_path, 'predlabels_seg.npy')
                        # with open(file_path, 'wb') as f:
                        #     np.save(f, predlabels)
                        # file_path = os.path.join(folder_path, 'GTlabels_seg.npy')
                        # with open(file_path, 'wb') as f:
                        #     np.save(f, GTlabels)
                        # file_path = os.path.join(folder_path, 'GTimus_seg.npy')
                        # with open(file_path, 'wb') as f:
                        #     np.save(f, GTimus)  # [left acc3, gyro3, right acc3, gyro3] = 12
                        # file_path = os.path.join(folder_path, 'loss_seg.npy')
                        # with open(file_path, 'wb') as f:
                        #     np.save(f, loss_list)
    print("average f1 score is: %f" % np.average(f1_scores))

if __name__ == "__main__":
    main()
