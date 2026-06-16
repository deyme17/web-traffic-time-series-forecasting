import torch
import torch.nn as nn
from .attention import build_attention



class BaselineLSTM(nn.Module):
    """Seq-to-seq LSTM RNN for web traffic time series forecasting."""
    def __init__(self, 
                 enc_in_size: int, 
                 dec_in_size: int, 
                 enc_h_size: int = 256, 
                 dec_h_size: int = 256, 
                 n_layers: int = 1, 
                 horizon: int = 365, 
                 attn_type: str = "none",
                 attn_size: int = 128,
                 attn_n_heads: int = 4,
                 dropout_enc: float = 0.,
                 dropout_dec: float = 0.,
                 dropout_ctx: float = 0.,
                 dropout_in: float = 0.,
                 dropout_h: float = 0.,
                 dropout_c: float = 0.,
                 dropout_out: float = 0.,
                 **kwargs) -> None:
        """
        Args:
            enc/dec_in_size: Number of features at each timestep for encoder/decoder.
            enc/dec_h_size: Hidden state size of encoder/decoder.
            n_layers: Number of stacked layers in the encoder and decoder LSTM.
            horizon: Number of future timesteps to predict.
            attn_type: Type of attention mechanism to use ('none' | 'additive' | 'dot' | 'multihead').
            attn_size: Projection size inside the attention module.
            attn_n_heads: Number of heads (for multihead attention only).
            dropout_enc: Dropout probability applied after each encoder layer.
            dropout_dec: Dropout probability applied after each decoder layer.
            dropout_ctx: Dropout for encoder's output that goes to the decoder.
            dropout_in: Dropout probability applied before decoder input layer.
            dropout_h: Dropout probability applied to decoder hidden state.
            dropout_c: Dropout probability applied to decoder c state.
            dropout_out: Dropout probability applied before each decoder output layer.
        """
        super().__init__()
        assert attn_type in ("none", "additive", "dot", "multihead")
        self.enc_in_size = enc_in_size
        self.dec_in_size = dec_in_size
        self.enc_h_size = enc_h_size
        self.dec_h_size = dec_h_size
        self.n_layers = n_layers
        self.horizon = horizon
        self.attn_type = attn_type
        self.attn_size = attn_size
        self.attn_n_heads = attn_n_heads

        self.h_states: torch.Tensor|None = None
        self.c_states: torch.Tensor|None = None

        # encoder
        self.encoder = nn.LSTM(
            input_size=enc_in_size,
            hidden_size=enc_h_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_enc if n_layers > 1 else 0.,
        )
        self.dropout_ctx = nn.Dropout(dropout_ctx)
        
        # attention
        self.attention = build_attention(
            attn_type=attn_type,
            query_size=dec_h_size,
            key_size=enc_h_size,
            attn_size=attn_size,
            n_heads=attn_n_heads
        )

        if attn_type == "none":
            context_size = enc_h_size * 2               # enc_context
        else:
            context_size = enc_h_size + enc_h_size * 2  # attention + enc_context

        # decoder
        self.dropout_in = nn.Dropout(dropout_in)
        self.dropout_dec = nn.Dropout(dropout_dec)
        self.dropout_h = nn.Dropout(dropout_h)
        self.dropout_c = nn.Dropout(dropout_c)
        self.dropout_out = nn.Dropout(dropout_out)

        self.decoder_in = nn.LSTMCell(
            dec_in_size + context_size + 1, 
            dec_h_size
        )
        self.decoder = nn.ModuleList([
            nn.LSTMCell(dec_h_size, dec_h_size)
            for _ in range(n_layers - 1)
        ])

        # out
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
        
        # decoder
        dec_h_states, dec_c_states = [], []
        preds = []
        prev_pred = torch.zeros(enc_in.shape[0], 1, device=enc_in.device)

        for t in range(self.horizon):
            # apply attention if provided
            if self.attention is not None:
                attn = self.attention(
                    query=h[0],
                    keys=enc_states
                )
                context = torch.cat([attn, enc_context], axis=1)
            else:
                context = enc_context

            x = torch.cat([prev_pred, dec_in[:, t, :], context], dim=-1)
            x = self.dropout_in(x)
            h[0], c[0] = self.decoder_in(x, (self.dropout_h(h[0]), 
                                             self.dropout_c(c[0])))
            out = h[0]

            for i, dec_cell in enumerate(self.decoder):
                out = self.dropout_dec(out)
                h[i + 1], c[i + 1] = dec_cell(out, (self.dropout_h(h[i + 1]), 
                                                    self.dropout_c(c[i + 1])))
                out = h[i + 1]

            dec_h_states.append(h[-1])
            dec_c_states.append(c[-1])

            pred = self.out_proj(self.dropout_out(out))
            preds.append(pred)
            prev_pred = pred

        self.h_states = torch.stack(dec_h_states, dim=1)
        self.c_states = torch.stack(dec_c_states, dim=1)

        return torch.cat(preds, dim=-1)