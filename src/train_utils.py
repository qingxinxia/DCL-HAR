import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics
import seaborn as sns
from src.utils import mds, tsne
import matplotlib.pyplot as plt


def train_time_series_seg(net, opt, criterion, train_loader,
                          device='cuda:0', batch_size=100):

    train_losses = []
    net.train()
    total, correct = 0, 0
    for i, (imus, labels) in enumerate(tqdm(train_loader)):
        imus = imus.to(device=device, non_blocking=True, dtype=torch.float)
        targets = labels.to(device=device, non_blocking=True, dtype=torch.int)

        # reshape input
        inputs = torch.transpose(imus.unsqueeze(3), 2, 1)
        output = net(inputs)

        loss = criterion(output.reshape(-1, output.shape[-1]),
                         targets.reshape(-1).long())
        # zero accumulated gradients
        opt.zero_grad()
        train_losses.append(loss.item())
        loss.backward()
        opt.step()

        # calculate accuracy
        target = targets.reshape(-1).long()
        output = output.reshape(-1, output.shape[-1])
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum()
    acc_test = float(correct) * 100.0 / total

    # print("Epoch: {}/{}...".format(e + 1, epochs),
    #       "Train Loss: {:.4f}...".format(np.mean(train_losses)))
    # "Val Loss: {:.4f}...".format(np.mean(val_losses)),
    # "Val Acc: {:.4f}...".format(accuracy / (len(X_test) // batch_size)),
    # "F1-Score: {:.4f}...".format(f1score / (len(X_test) // batch_size)))
    return np.mean(train_losses), acc_test


def eval_time_series_seg(net, criterion, test_loader, device='cuda:0', batch_size=100):
    # val_h = net.init_hidden(batch_size)
    val_losses = []
    accuracy = 0
    f1score = 0
    net.eval()
    GTlabel_list, predlabel_list, GTimu_list = [], [], []
    with torch.no_grad():
        for i, (imus, labels) in enumerate(tqdm(test_loader)):
            inputs = imus.to(device=device, non_blocking=True, dtype=torch.float)
            targets = labels.to(device=device, non_blocking=True, dtype=torch.int)

            # val_h = tuple([each.data for each in val_h])

            # output, val_h = net(inputs, val_h, batch_size)
            output, _ = net(inputs)

            val_loss = criterion(output, targets[:, 0, :].reshape(-1).long())
            val_losses.append(val_loss.item())

            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets[:, 0, :].reshape(-1).view(*top_class.shape).long()
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            f1score += metrics.f1_score(top_class.cpu(),
                                        targets[:, 0, :].reshape(-1).view(*top_class.shape).long().cpu(),
                                        average='weighted')

            tmp_imu = imus.detach().cpu().numpy()
            GTimu_list.append(tmp_imu)
            tmp_label = labels.detach().cpu().numpy()
            GTlabel_list.append(tmp_label[:, :, 0])
            tmp_label = top_class.detach().cpu().numpy()
            predlabel_list.append(np.tile(tmp_label, (1, labels.shape[1])))

    datalength = test_loader.batch_sampler.sampler.data_source.label.data.shape[0]
    print("Val Loss: {:.4f}...".format(np.mean(val_losses)),
          "Val Acc: {:.4f}...".format(accuracy / (datalength // batch_size)),
          "F1-Score: {:.4f}...".format(f1score / (datalength // batch_size)))

    return GTimu_list, GTlabel_list, predlabel_list


def test(test_loader, model, DEVICE, criterion, n_class, folder_path, epoch, plts=False, savef=True):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 0
        total = 0
        correct = 0
        feats = None
        prds = None
        trgs = None
        confusion_matrix = torch.zeros(n_class, n_class)
        GTlabel_list, predlabel_list, GTimu_list = [], [], []
        for idx, (sample, target) in enumerate(test_loader):
            n_batches += 1
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()

            # reshape input
            sample = torch.transpose(sample.unsqueeze(3), 2, 1)
            out = model(sample)
            # reshape output
            target = target.reshape(-1).long()
            output = out.reshape(-1, out.shape[-1])
            loss = criterion(output, target)
            # loss = criterion(out, target[:, 0, :].reshape(-1).long())
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()

            if prds is None:
                prds = predicted
                trgs = target
                # feats = features[:, :]
            else:
                prds = torch.cat((prds, predicted))
                trgs = torch.cat((trgs, target))
                # feats = torch.cat((feats, features), 0)

            # save data
            if savef == True:
                tmp_imu = sample.detach().cpu().numpy()
                GTimu_list.append(tmp_imu)

                tmp_label = target.detach().cpu().numpy()
                tmp_label = tmp_label.reshape(sample.shape[0], -1)
                GTlabel_list.append(tmp_label)

                tmp_label = predicted.detach().cpu().numpy()
                tmp_label = tmp_label.reshape(sample.shape[0], -1)
                predlabel_list.append(tmp_label)
            # predlabel_list.append(np.tile(tmp_label, (1, target.shape[1])))

        acc_test = float(correct) * 100.0 / total

    print({"dev": {"Test Loss": total_loss / n_batches}})
    print({"dev": {"Test Acc": acc_test}})

    print(f'Test Loss     : {total_loss / n_batches:.4f}\t | \tTest Accuracy     : {acc_test:2.4f}\n')
    for t, p in zip(trgs.view(-1), prds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    print(confusion_matrix)
    print(confusion_matrix.diag() / confusion_matrix.sum(1))
    # print(confusion_matrix, name='conf_mat')
    if plts == True:
        plt.figure(figsize=(10, 8))
        # tsne(feats.detach().cpu().numpy(), trgs.detach().cpu().numpy(), save_dir=os.path.join(folder_path, 'tsne_epoch_%s.png'%str(epoch)))
        # mds(feats.detach().cpu().numpy(), trgs.detach().cpu().numpy(), save_dir=os.path.join(folder_path, 'mds_epoch_%s.png'%str(epoch)))
        sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(os.path.join(folder_path, 'confmatrix_epoch_%s.png' % str(epoch)))
        plt.close()
    if savef == True:
        return GTimu_list, GTlabel_list, predlabel_list
    else:
        return
