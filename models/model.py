import torch
import torch.nn as nn
from utils import Registry


MODELS = Registry()



@MODELS.register("LSTM")
class WTTSF_LSTM(nn.Module):
    """Seq-to-seq LSTM RNN for web traffic time series forecasting."""
    def __init__(self, 
                 enc_in_size: int, 
                 dec_in_size: int, 
                 enc_h_size: int = 256, 
                 dec_h_size: int = 256, 
                 n_layers: int = 1, 
                 horizon: int = 365, 
                 dropout: float = 0., 
                 dropout_cntx: float = 0.) -> None:
        """
        Args:
            enc/dec_in_size: Number of features at each timestep for encoder/decoder.
            enc/dec_h_size: Hidden state size of encoder/decoder.
            n_layers: Number of stacked layers in the encoder and decoder LSTM.
            horizon: Number of future timesteps to predict.
            dropout: Dropout probability applied after each decoder layer output.
            dropout_cntx: Dropout for encoder's output (context) that goes to the decoder.
        """
        super().__init__()
        self.enc_in_size = enc_in_size
        self.dec_in_size = dec_in_size
        self.enc_h_size = enc_h_size
        self.dec_h_size = dec_h_size
        self.n_layers = n_layers
        self.horizon = horizon

        # encoder
        self.encoder = nn.LSTM(
            input_size=enc_in_size,
            hidden_size=enc_h_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.,
        )
        self.dropout_cntx = nn.Dropout(dropout_cntx)

        # decoder
        self.decoder_in = nn.LSTMCell(dec_in_size + enc_h_size * 2, dec_h_size)
        self.decoder = nn.ModuleList([
            nn.LSTMCell(dec_h_size, dec_h_size)
            for _ in range(n_layers - 1)
        ])

        # out
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dec_h_size, 1)

    def forward(self, enc_in: torch.Tensor, dec_in: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model. Returns prediction with size (batch_size, horizon)."""
        # encoder
        enc_state, (h_n, c_n) = self.encoder(enc_in)
        enc_last = enc_state[:, -1, :]
        enc_mean = enc_state.mean(dim=1)
        context = self.dropout_cntx(torch.cat([enc_last, enc_mean], dim=-1))

        h = [h_n[i] for i in range(self.n_layers)]
        c = [c_n[i] for i in range(self.n_layers)]
        
        # decoder
        preds = []
        for t in range(self.horizon):

            x = torch.cat([dec_in[:, t, :], context], dim=-1)
            h[0], c[0] = self.decoder_in(x, (h[0], c[0]))
            out = h[0]

            for i, dec_cell in enumerate(self.decoder):
                h[i + 1], c[i + 1] = dec_cell(out, (h[i + 1], c[i + 1]))
                out = h[i + 1]

            preds.append(self.out_proj(self.dropout(out)))

        return torch.cat(preds, dim=-1)