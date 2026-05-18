import math

import torch
import torch.nn as nn
import torch.nn.init as init


class LipNetGRU(nn.Module):
    """
    Original LipNet architecture: STConv3D frontend + 2x Bi-GRU + CTC.
    Input:  (B, 3, T, 64, 128)
    Output: (B, T, num_classes)
    """

    def __init__(self, dropout_p: float = 0.5, num_classes: int = 28):
        super().__init__()

        # --- STConv frontend ---
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        # After STConv: (B, 96, T, 4, 8) → flat dim = 96*4*8 = 3072
        self.gru1 = nn.GRU(96 * 4 * 8, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.FC = nn.Linear(512, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout3d = nn.Dropout3d(dropout_p)

        self._init_weights()

    def _init_weights(self):
        for conv in (self.conv1, self.conv2, self.conv3):
            init.kaiming_normal_(conv.weight, nonlinearity='relu')
            init.constant_(conv.bias, 0)

        init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        init.constant_(self.FC.bias, 0)

        stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
        for gru in (self.gru1, self.gru2):
            for i in range(0, 256 * 3, 256):
                init.uniform_(gru.weight_ih_l0[i:i + 256],
                               -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(gru.weight_hh_l0[i:i + 256])
                init.constant_(gru.bias_ih_l0[i:i + 256], 0)
                init.uniform_(gru.weight_ih_l0_reverse[i:i + 256],
                               -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(gru.weight_hh_l0_reverse[i:i + 256])
                init.constant_(gru.bias_ih_l0_reverse[i:i + 256], 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, 64, 128)
        x = self.pool1(self.dropout3d(self.relu(self.conv1(x))))
        x = self.pool2(self.dropout3d(self.relu(self.conv2(x))))
        x = self.pool3(self.dropout3d(self.relu(self.conv3(x))))

        # (B, C, T, H, W) → (T, B, C*H*W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()

        x, _ = self.gru1(x)
        x = self.dropout(x)
        x, _ = self.gru2(x)
        x = self.dropout(x)

        x = self.FC(x)                      # (T, B, num_classes)
        return x.permute(1, 0, 2).contiguous()  # (B, T, num_classes)
