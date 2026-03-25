import torch
import torch.nn as nn

from bus_tt.constants import CTX_DIM


class PhyLSTMModel(nn.Module):
    """
    Shared architecture for both LSTM and Phy-LSTM.
    The difference is the loss function used during training:
    - LSTM uses MSE only.
    - Phy-LSTM uses MSE + physics-informed loss.

    Architecture: LSTM -> LayerNorm -> concat(h, x_ctx) -> FFN -> output.
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.2, ctx_dim: int = CTX_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

        concat_dim = hidden_dim + ctx_dim
        self.ffn1 = nn.Linear(concat_dim, concat_dim // 2)
        self.ffn2 = nn.Linear(concat_dim // 2, concat_dim // 4)
        self.out = nn.Linear(concat_dim // 4, 1)
        self.act = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_seq, x_ctx):
        lstm_out, _ = self.lstm(x_seq)
        h = self.drop(self.norm(lstm_out[:, -1, :]))
        x = self.drop(self.act(self.ffn1(torch.cat([h, x_ctx], dim=1))))
        x = self.drop(self.act(self.ffn2(x)))
        return self.out(x)
