import math

import torch
import torch.nn as nn
import torch.nn.init as init


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in "Attention Is All You Need".
    Expects input of shape (T, B, d_model) and adds non-learnable position
    signals so the Transformer can exploit temporal order.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                       # (L, d)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)                                     # (L, 1, d) — broadcasts over B
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LipNetTransformer(nn.Module):
    """
    LipNet variant: STConv3D frontend (identical to original) replaced
    Bi-GRU sequence model with a Transformer Encoder + sinusoidal PE + CTC.

    Architecture:
        Input  → STConv (3 × Conv3D + MaxPool3D) → flatten spatial dims
               → Linear projection to d_model
               → SinusoidalPositionalEncoding
               → nn.TransformerEncoder (num_layers × EncoderLayer)
               → Linear → (B, T, num_classes)

    Input:  (B, 3, T, 64, 128)
    Output: (B, T, num_classes)
    """

    def __init__(
        self,
        dropout_p: float = 0.5,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        attn_dropout: float = 0.1,
        num_classes: int = 28,
    ):
        super().__init__()

        # --- STConv frontend (same as LipNetGRU) ---
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        # After STConv: (B, 96, T, 4, 8) → flat = 96*4*8 = 3072
        conv_out_dim = 96 * 4 * 8

        # --- Sequence model: Transformer ---
        self.input_proj = nn.Linear(conv_out_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=attn_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=attn_dropout,
            batch_first=False,   # expects (T, B, d_model)
            norm_first=True,     # pre-LN is more stable for speech/video tasks
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.FC = nn.Linear(d_model, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout3d = nn.Dropout3d(dropout_p)

        self._init_weights()

    def _init_weights(self):
        for conv in (self.conv1, self.conv2, self.conv3):
            init.kaiming_normal_(conv.weight, nonlinearity='relu')
            init.constant_(conv.bias, 0)

        init.xavier_uniform_(self.input_proj.weight)
        init.constant_(self.input_proj.bias, 0)

        init.xavier_uniform_(self.FC.weight)
        init.constant_(self.FC.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, 64, 128)
        x = self.pool1(self.dropout3d(self.relu(self.conv1(x))))
        x = self.pool2(self.dropout3d(self.relu(self.conv2(x))))
        x = self.pool3(self.dropout3d(self.relu(self.conv3(x))))

        # (B, C, T, H, W) → (T, B, C*H*W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        x = self.input_proj(x)      # (T, B, d_model)
        x = self.pos_enc(x)         # + sinusoidal PE

        x = self.transformer_encoder(x)   # (T, B, d_model)

        x = self.FC(x)                         # (T, B, num_classes)
        return x.permute(1, 0, 2).contiguous()  # (B, T, num_classes)
