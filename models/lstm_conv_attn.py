import torch
import torch.nn as nn
from .attention.conv_attention import ConvAttention



class ConvAttnLSTM(nn.Module):
    """Seq-to-seq LSTM RNN with ConvAttention for web traffic time series forecasting."""
    def __init__(self, 
                 enc_in_size: int,
                 dec_in_size: int,
                 enc_h_size: int = 256,
                 dec_h_size: int = 256,
                 n_layers: int = 1,
                 horizon: int = 62,
                 lookback: int = 365,
                 dropout: float = 0.,
                 dropout_ctx: float = 0.,
                 readout_size: int = 128,
                 fingerprint_size: int = 16,
                 attn_n_heads: int = 4) -> None:
        """
        Args:
            enc/dec_in_size: Number of features at each timestep for encoder/decoder.
            enc/dec_h_size: Hidden state size of encoder/decoder.
            n_layers: Number of stacked layers in the encoder and decoder LSTM.
            horizon: Number of future timesteps to predict.
            lookback: Number of past timesteps for attention window.
            dropout: Dropout probability applied after each decoder layer output.
            dropout_ctx: Dropout for encoder's output (context) that goes to the decoder.
            readout_size: Depth of compressed enc_states readout in ConvAttention
            fingerprint_size: Size of the 'fingerprint' given by ConvFingerprint.
            attn_n_heads: Number of attention heads in ConvAttention.
        """
        super().__init__()
        self.enc_in_size = enc_in_size
        self.dec_in_size = dec_in_size
        self.enc_h_size = enc_h_size
        self.dec_h_size = dec_h_size
        self.n_layers = n_layers
        self.horizon = horizon
        self.attn_window = lookback - horizon + 1
        self.readout_size = readout_size
        self.fingerprint_size = fingerprint_size
        self.attn_n_heads = attn_n_heads

        # encoder
        self.encoder = nn.LSTM(
            input_size=enc_in_size,
            hidden_size=enc_h_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.
        )
        self.dropout_ctx = nn.Dropout(dropout_ctx)

        # attention
        self.conv_attn = ConvAttention(
            enc_in_size=enc_in_size,
            enc_h_size=enc_h_size,
            readout_size=readout_size,
            fingerprint_size=fingerprint_size,
            attn_window=self.attn_window,
            horizon=horizon,
            n_heads=attn_n_heads
        )

        context_size = enc_h_size * 2  # enc_last + enc_mean
        attn_out_size = readout_size * attn_n_heads

        # decoder
        self.decoder_in = nn.LSTMCell(
            dec_in_size + context_size + attn_out_size + 1, 
            dec_h_size
        )
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
        enc_states, (h_n, c_n) = self.encoder(enc_in)
        enc_last = enc_states[:, -1, :]
        enc_mean = enc_states.mean(dim=1)
        enc_context = self.dropout_ctx(torch.cat([enc_last, enc_mean], dim=-1))

        h = [h_n[i] for i in range(self.n_layers)]
        c = [c_n[i] for i in range(self.n_layers)]

        # attention
        attn = self.conv_attn(enc_in, enc_states)

        # decoder
        preds = []
        prev_pred = torch.zeros(enc_in.size(0), 1, device=enc_in.device)

        for t in range(self.horizon):
            x = torch.cat([prev_pred, dec_in[:, t, :], attn[:, t, :], enc_context], dim=-1)
            h[0], c[0] = self.decoder_in(x, (h[0], c[0]))
            out = h[0]

            for i, dec_cell in enumerate(self.decoder):
                h[i + 1], c[i + 1] = dec_cell(out, (h[i + 1], c[i + 1]))
                out = h[i + 1]

            pred = self.out_proj(self.dropout(out))
            preds.append(pred)
            prev_pred = pred.detach()

        return torch.cat(preds, dim=-1)