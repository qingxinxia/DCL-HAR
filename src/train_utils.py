
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics

def train_time_series_seg(net, opt, criterion, train_loader,
                          device='cuda:0', batch_size=100):
    # opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # criterion = nn.CrossEntropyLoss()
    # losses = []

    # initialize hidden state
    h = net.init_hidden(batch_size)
    train_losses = []
    net.train()
    for i, (imus, labels) in enumerate(tqdm(train_loader)):
        inputs = imus.to(device=device, non_blocking=True, dtype=torch.float)
        targets = labels.to(device=device, non_blocking=True, dtype=torch.int)

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        opt.zero_grad()

        # get the output from the model
        output, h = net(inputs, h, batch_size)

        loss = criterion(output, targets[:,0,:].reshape(-1).long())
        train_losses.append(loss.item())
        loss.backward()
        opt.step()

    # print("Epoch: {}/{}...".format(e + 1, epochs),
    #       "Train Loss: {:.4f}...".format(np.mean(train_losses)))
          # "Val Loss: {:.4f}...".format(np.mean(val_losses)),
          # "Val Acc: {:.4f}...".format(accuracy / (len(X_test) // batch_size)),
          # "F1-Score: {:.4f}...".format(f1score / (len(X_test) // batch_size)))
    return np.mean(train_losses)


def eval_time_series_seg(net, criterion, test_loader, device='cuda:0', batch_size=100):
    val_h = net.init_hidden(batch_size)
    val_losses = []
    accuracy = 0
    f1score = 0
    net.eval()
    GTlabel_list, predlabel_list, GTimu_list = [], [], []
    with torch.no_grad():
        for i, (imus, labels) in enumerate(tqdm(test_loader)):
            inputs = imus.to(device=device, non_blocking=True, dtype=torch.float)
            targets = labels.to(device=device, non_blocking=True, dtype=torch.int)

            val_h = tuple([each.data for each in val_h])

            output, val_h = net(inputs, val_h, batch_size)

            val_loss = criterion(output, targets[:,0,:].reshape(-1).long())
            val_losses.append(val_loss.item())

            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets[:,0,:].reshape(-1).view(*top_class.shape).long()
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            f1score += metrics.f1_score(top_class.cpu(),
                                        targets[:,0,:].reshape(-1).view(*top_class.shape).long().cpu(),
                                        average='weighted')

            tmp_imu = imus.detach().cpu().numpy()
            GTimu_list.append(tmp_imu)
            tmp_label = labels.detach().cpu().numpy()
            GTlabel_list.append(tmp_label[:, :, 0])
            tmp_label = top_class.detach().cpu().numpy()
            predlabel_list.append(np.tile(tmp_label, (1, labels.shape[1])))

    # print("Val Loss: {:.4f}...".format(np.mean(val_losses)),
    #       "Val Acc: {:.4f}...".format(accuracy / (len(X_test) // batch_size)),
    #       "F1-Score: {:.4f}...".format(f1score / (len(X_test) // batch_size)))

    return GTimu_list, GTlabel_list, predlabel_list