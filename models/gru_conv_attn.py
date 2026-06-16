import torch
import torch.nn as nn
from .attention.conv_attention import ConvAttention



class ConvAttnGRU(nn.Module):
    """Seq-to-seq GRU RNN with ConvAttention for web traffic time series forecasting."""
    def __init__(self, 
                 enc_in_size: int,
                 dec_in_size: int,
                 enc_h_size: int = 256,
                 dec_h_size: int = 256,
                 n_layers: int = 1,
                 horizon: int = 62,
                 lookback: int = 365,
                 readout_size: int = 128,
                 fingerprint_size: int = 16,
                 attn_n_heads: int = 4,
                 dropout_enc: float = 0.,
                 dropout_in: float = 0.,
                 dropout_h: float = 0.,
                 dropout_out: float = 0.,
                 dropout_ctx: float = 0.,
                 readout_dropout: float = 0.,
                 fingerprint_dropout: float = 0.,
                 **kwargs) -> None:
        """
        Args:
            enc/dec_in_size: Number of features at each timestep for encoder/decoder.
            enc/dec_h_size: Hidden state size of encoder/decoder.
            n_layers: Number of stacked layers in encoder and decoder LSTM.
            horizon: Number of future timesteps to predict.
            lookback: Number of past timesteps for attention window.
            readout_size: Compressed readout depth in ConvAttention.
            fingerprint_size: CNN output size for ConvFingerprint.
            attn_n_heads: Number of attention heads in ConvAttention.
            dropout_enc: Dropout probability applied after each encoder layer.
            dropout_in: Dropout probability applied after decoder input layer.
            dropout_h: Dropout probability applied after each decoder hidden layer.
            dropout_out: Dropout probability applied after each decoder output layer.
            dropout_ctx: Dropout on encoder hidden state passed to decoder init.
            readout_dropout: Dropout before readout projection in ConvAttention.
            fingerprint_dropout: Dropout before ConvFingerprint fully connected layer.
        """
        super().__init__()
        self.enc_in_size = enc_in_size
        self.dec_in_size = dec_in_size
        self.enc_h_size = enc_h_size
        self.dec_h_size = dec_h_size
        self.n_layers = n_layers
        self.horizon = horizon
        self.lookback = lookback
        self.attn_window = lookback - horizon + 1
        self.readout_size = readout_size
        self.fingerprint_size = fingerprint_size
        self.attn_n_heads = attn_n_heads

        # encoder
        self.encoder = nn.GRU(
            input_size=enc_in_size,
            hidden_size=enc_h_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_enc if n_layers > 1 else 0.
        )

        # enc-dec transition
        self.enc_to_dec_h = nn.ModuleList([
            nn.Linear(enc_h_size, dec_h_size) for _ in range(n_layers)
        ])
        self.dropout_ctx = nn.Dropout(dropout_ctx)

        # attention
        self.conv_attn = ConvAttention(
            enc_h_size=enc_h_size,
            readout_size=readout_size,
            fingerprint_size=fingerprint_size,
            attn_window=self.attn_window,
            horizon=horizon,
            lookback=lookback,
            n_heads=attn_n_heads,
            readout_dropout=readout_dropout,
            fingerprint_dropout=fingerprint_dropout,
        )
        attn_out_size = readout_size * attn_n_heads

        # decoder
        self.dropout_in = nn.Dropout(dropout_in)
        self.decoder_in = nn.GRUCell(
            1 + dec_in_size + attn_out_size,
            dec_h_size
        )
        self.dropout_h = nn.Dropout(dropout_h)
        self.decoder = nn.ModuleList([
            nn.GRUCell(dec_h_size, dec_h_size)
            for _ in range(n_layers - 1)
        ])

        # out
        self.dropout_out = nn.Dropout(dropout_out)
        self.out_proj = nn.Linear(dec_h_size, 1)

    def _init_decoder_state(self, h_n: torch.Tensor) -> list[torch.Tensor]:
        """Project encoder final states into decoder hidden states."""
        h = []
        for i in range(self.n_layers):
            hi = self.dropout_ctx(h_n[i])
            h.append(torch.tanh(self.enc_to_dec_h[i](hi)))
        return h

    def forward(self, enc_in: torch.Tensor, dec_in: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model. Returns prediction with size (batch_size, horizon)."""
        # encoder
        enc_states, h_n = self.encoder(enc_in)

        # attention
        attn = self.conv_attn(enc_in, enc_states)

        # decoder
        h = self._init_decoder_state(h_n)

        preds = []
        prev_pred = torch.zeros(enc_in.size(0), 1, device=enc_in.device)

        for t in range(self.horizon):
            x = torch.cat([prev_pred, dec_in[:, t, :], attn[:, t, :]], dim=-1)
            x = self.dropout_in(x)
            h[0] = self.decoder_in(x, h[0])
            out = h[0]

            for i, dec_cell in enumerate(self.decoder):
                out = self.dropout_h(out)
                h[i + 1] = dec_cell(out, h[i + 1])
                out = h[i + 1]

            pred = self.out_proj(self.dropout_out(out))
            preds.append(pred)
            prev_pred = pred

        return torch.cat(preds, dim=-1)