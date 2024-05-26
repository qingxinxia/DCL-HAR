import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
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
        self.lstm6 = nn.LSTM(num_conv_filter,
                             hidden_units,
                             batch_first=True,
                             bidirectional=True)
        self.lstm7 = nn.LSTM(hidden_units*2,
                             hidden_units,
                             batch_first=True,
                             bidirectional=True)
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


class DeepConvLSTMSelfAttn(nn.Module):  # DCLSA
    """
    Implementation of a DeepConvLSTM with Self-Attention used in
    'Deep ConvLSTM with self-attention for human activity decoding
    using wearable sensors' (Sensors 2020).

    Note:
        https://ieeexplore.ieee.org/document/9296308 (Sensors 2020)

    """

    def __init__(self, in_ch: int = 3, num_classes: int = 11):
        super().__init__()

        # *** Edit Here ***
        # NOTE: Experiment 1-A: Kernel Size (NOTE: set odd number)
        ks = 5
        # NOTE: Experiment 1-B: # of convolutional layers (Default: 4)
        self.num_conv_layers = 4
        self.num_attn_layers = 2
        self.num_attn_heads = 2
        # ******************
        num_conv_filter = 64  # convolutional filters (Default: 64)

        # -- [1] Embedding Layer --
        # Convolutions
        conv_blocks = []
        for i in range(self.num_conv_layers):
            in_ch_ = in_ch if i == 0 else 64
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch_,
                              64,
                              kernel_size=(5, 1),
                              padding=(2, 0)),  # 5 // 2 → 2
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
            )
        self.conv_blocks = nn.ModuleList(conv_blocks)

        # # Added for experiment 06 model params (hyperparameter tuning)
        # if self.dropout_after_conv == True:
        #     self.conv_dropout_layer = nn.Dropout(p=0.5)

        # -- [2] LSTM Encoder -
        # # *** Edit Here ***
        hidden_units = 128  # number of hidden units for Bi-LSTM
        # ******************-
        self.lstm6 = nn.LSTM(num_conv_filter,
                             hidden_units,
                             batch_first=True,
                             bidirectional=True)
        self.lstm7 = nn.LSTM(hidden_units * 2,
                             hidden_units,
                             batch_first=True,
                             bidirectional=True)
        self.dropout6 = nn.Dropout(p=0.3)
        self.dropout7 = nn.Dropout(p=0.3)

        # dropout_blocks = []
        # for i in range(self.num_lstm_layers):
        #     dropout_blocks.append(
        #         nn.Sequential(
        #             nn.Dropout(p=0.5)
        #         )
        #     )
        # self.dropout_blocks = nn.ModuleList(dropout_blocks)

        # -- [3] Self-Attention --
        attn_blocks = []
        for i in range(self.num_attn_layers):
            attn_blocks.append(
                nn.MultiheadAttention(hidden_units * 2, #self.output_lstm_units,
                                      self.num_attn_heads,
                                      batch_first=True, )
            )
        self.attn_blocks = nn.ModuleList(attn_blocks)

        # -- [4] Output Layer --
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.out = nn.Conv2d(hidden_units * 2,
                            num_classes,
                            1,
                            stride=1,
                            padding=0,)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (B, CH, T, 1)
        Returns:
            torch.Tensor: shape = (B, N_CLASSES, T, 1)
        """
        # -- [1] Embedding Layer --
        # Convolutions
        for i in range(self.num_conv_layers):
            x = self.conv_blocks[i](x)

        # # Added for experiment 06 model params (hyperparameter tuning)
        # if self.dropout_after_conv == True:
        #     x = self.conv_dropout_layer(x)

        # -- [2] LSTM Encoder --
        # Reshape: (B, CH, T, 1) -> (B, T, CH)
        # nn.LSTM(): batch_first=True
        x = x.squeeze(3).transpose(1, 2)
        x, _ = self.lstm6(x)
        x = self.dropout6(x)
        x, _ = self.lstm7(x)
        x = self.dropout7(x)

        # -- [3] Self-Attention --
        for i in range(self.num_attn_layers):
            x, w = self.attn_blocks[i](x.clone(), x.clone(), x.clone())
        # Reshape: (B, T, CH) -> (B, CH, T, 1)
        x = x.transpose(1, 2).unsqueeze(3)

        # -- [4] Output Layer --

        x = self.out(x)
        x_out = torch.transpose(x.squeeze(-1), 2, 1)  # output.shape=(batch,datalen,cls)

        return x_out