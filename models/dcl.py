import torch
import torch.nn as nn
import torch.nn.functional as F

# class HARModel(nn.Module):
#
#     def __init__(self, in_feature, n_hidden=128, n_layers=1,
#                  n_filters=64, n_classes=18, filter_size=5,
#                  datalen=60, drop_prob=0.5, device='cuda:0'):
#         super(HARModel, self).__init__()
#         self.num_sensor_channels = in_feature
#         self.drop_prob = drop_prob
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.n_filters = n_filters
#         self.n_classes = n_classes
#         self.filter_size = filter_size
#         self.device = device
#         self.datalen = datalen
#
#         self.conv1 = nn.Conv1d(in_feature, n_filters, filter_size)
#         self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
#         self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
#         self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)
#
#         self.lstm1 = nn.LSTM(n_filters, n_hidden, n_layers)
#         self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers)
#
#         self.fc = nn.Linear(n_hidden, n_classes)
#
#         self.dropout = nn.Dropout(drop_prob)
#
#     def forward(self, input, hidden, batch_size):
#         # input = (batch, datalen, dim)
#         x = input.view(-1, self.num_sensor_channels, self.datalen)
#         # x = x.view(-1, self.num_sensor_channels, SLIDING_WINDOW_LENGTH)
#         x = F.relu(self.conv1(x))  # input x=()
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#
#         # x = x.view(2, -1, self.n_filters)  # the first dimension=hidden[0].shape[-1]/self.n_filters
#         x = x.view(-1, hidden[0].shape[1], self.n_filters)
#         # x = x.view(8, -1, self.n_filters)
#         x, hidden = self.lstm1(x, hidden)
#         x, hidden = self.lstm2(x, hidden)
#
#         x = x.contiguous().view(-1, self.n_hidden)
#         x = self.dropout(x)
#         x = self.fc(x)
#
#         out = x.view(batch_size, -1, self.n_classes)[:, -1, :]  # out shape=(batch, num_classes)
#
#         return out, hidden
#
#     def init_hidden(self, batch_size):
#         ''' Initializes hidden state '''
#         # Create two new tensors with sizes n_layers x batch_size x n_hidden,
#         # initialized to zero, for hidden state and cell state of LSTM
#         weight = next(self.parameters()).data
#
#         # if (train_on_gpu):
#         hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device),
#                   weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device))
#                       # weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
#         # else:
#         #     hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
#         #               weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
#
#         return hidden
#
#
# # net = HARModel()
#
# class DeepConvLSTM(nn.Module):
#     def __init__(self, n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True):
#         super(DeepConvLSTM, self).__init__()
#
#         self.backbone = backbone
#
#         self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
#         self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
#         self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
#         self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
#
#         self.dropout = nn.Dropout(0.5)
#         self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)
#
#         self.out_dim = LSTM_units
#
#         if backbone == False:
#             self.classifier = nn.Linear(LSTM_units, n_classes)
#
#         self.activation = nn.ReLU()
#
#     def forward(self, x):  # (batch, datalen, datadim)
#         self.lstm.flatten_parameters()
#         x = x.unsqueeze(1)  # out_x.shape=(batch, 1, datalen, datadim)
#         x = self.activation(self.conv1(x))  # out_x.shape=(batch, 64, datalen52, datadim)
#         x = self.activation(self.conv2(x))
#         x = self.activation(self.conv3(x))  # out_x.shape=(batch, 64, datalen48, datadim)
#         x = self.activation(self.conv4(x))  # out_x.shape=(batch, 64, datalen44, datadim)
#
#         x = x.permute(2, 0, 3, 1)
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # out_x.shape=(datalen44, batch, -1)
#
#         x = self.dropout(x)
#
#         x, h = self.lstm(x)  # out_x.shape=(datalen44, batch, 128)
#         x = x[-1, :, :]
#
#         if self.backbone:
#             return None, x
#         else:
#             out = self.classifier(x)  # out.shape=(512, cls), x.shape=(512, 128)
#             return out, x

class DeepConvLstmV1(nn.Module):
    """CNN Only
    """
    def __init__(self, in_ch: int = 3, num_classes: int = 11):
        super().__init__()
        # if num_classes is None:
        #     num_classes = len(OPENPACK_OPERATIONS)

        # -- [1] CNN  --
        # *** Edit Here ***
        # NOTE: Experiment 1-A: Kernel Size (NOTE: set odd number)
        ks = 5
        # NOTE: Experiment 1-B: # of convolutional layers (Default: 4)
        num_conv_layers = 4
        # ******************
        num_conv_filter = 64 # convolutional filters (Default: 64)

        blocks = []
        for i in range(num_conv_layers):
            in_ch_ = in_ch if i == 0 else num_conv_filter
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch_, num_conv_filter, kernel_size=(ks, 1), padding=(ks//2, 0)),
                    nn.BatchNorm2d(num_conv_filter),
                    nn.ReLU(),
                )
            )
        self.conv_blocks = nn.ModuleList(blocks)

        # -- [3] Output --
        self.out8 = nn.Conv2d(
            num_conv_filter,
            num_classes,
            1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (B, CH, T, 1)
        Returns:
            torch.Tensor: shape = (B, N_CLASSES, T, 1)
        """
        # -- [1] Conv --
        for block in self.conv_blocks:
            x = block(x)

        # -- [3] Output --
        x = self.out8(x)  # output.shape=(batch, cls, datalen, 1)
        x_out = torch.transpose(x.squeeze(-1), 2,1)  # output.shape=(batch,datalen,cls)
        return x_out

class DeepConvLstmV3(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 11):
        super().__init__()
        # if num_classes is None:
        #     num_classes = len(OPENPACK_OPERATIONS)

        # -- [1] CNN --
        # *** Edit Here ***
        num_conv_layers = 4 # convolutional layers (Default: 4)
        num_conv_filter = 64 # convolutional filters (Default: 64)
        ks = 5 # kernel size,
        # ******************

        blocks = []
        for i in range(num_conv_layers):
            in_ch_ = in_ch if i == 0 else 64
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch_, 64, kernel_size=(5, 1), padding=(2, 0)),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
            )
        self.conv_blocks = nn.ModuleList(blocks)

        # -- [2] LSTM --
        # *** Edit Here ***
        hidden_units = 128 # number of hidden units for Bi-LSTM
        # ******************

        # NOTE: enable ``bidirectional``
        self.lstm6 = nn.LSTM(num_conv_filter, hidden_units, batch_first=True, bidirectional=True)
        self.lstm7 = nn.LSTM(hidden_units*2, hidden_units, batch_first=True,  bidirectional=True)
        self.dropout6 = nn.Dropout(p=0.3)
        self.dropout7 = nn.Dropout(p=0.3)

        # -- [3] Output --
        self.out8 = nn.Conv2d(
            hidden_units * 2,
            num_classes,
            1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (B, CH, T, 1)
        Returns:
            torch.Tensor: shape = (B, N_CLASSES, T, 1)
        """
        # -- [1] Conv --
        for block in self.conv_blocks:
            x = block(x)

        # -- [2] LSTM --
        # Reshape: (B, CH, 1, T) -> (B, T, CH)
        x = x.squeeze(3).transpose(1, 2)

        x, _ = self.lstm6(x)
        x = self.dropout6(x)
        x, _ = self.lstm7(x)
        x = self.dropout7(x)

        # Reshape: (B, T, CH) -> (B, CH, T, 1)
        x = x.transpose(1, 2).unsqueeze(3)

        # -- [3] Output --
        x = self.out8(x)
        x_out = torch.transpose(x.squeeze(-1), 2, 1)  # output.shape=(batch,datalen,cls)

        return x_out