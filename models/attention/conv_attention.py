import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.constants import FINGERPRINT_SIGNAL



class ConvFingerprint(nn.Module):
    """CNN that produces a 'fingerprint' of the input timeseries."""
    def __init__(self, in_ch: int = 1, out_size: int = 16, 
                 lookback: int = 365, dropout: bool = 0.):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 * (lookback // 8), 512), nn.SELU(),
            nn.Linear(512, out_size), nn.SELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) # [B, T, <-> C]
        x = self.convnet(x)
        x = x.flatten(1)
        return self.fc(x)



class ConvAttention(nn.Module):
    """Attention on the encoder fingerprint given by depthwise conv."""
    def __init__(self,
                 enc_h_size: int,
                 readout_size: int,
                 fingerprint_size: int,
                 attn_window: int,
                 horizon: int,
                 lookback: int,
                 n_heads: int = 4,
                 readout_dropout: bool = 0.,
                 fingerprint_dropout: bool = 0.):
        """
        Args:
            enc_h_size: Encoder hidden size.
            readout_size: Compressed readout depth.
            fingerprint_size: CNN output size.
            attn_window: Lookback - horizon + 1.
            horizon: Prediction window.
            lookback: Past timestep size.
            n_heads: Number of attention heads.
            readout_dropout: Dropout before readout_proj.
            fingerprint_dropout: Dropout before ConvFingerprint.fc
        """
        super().__init__()
        self.n_heads = n_heads
        self.readout_size = readout_size
        self.horizon = horizon
        self.lookback = lookback
        self.attn_window = attn_window
        self.n_signal = FINGERPRINT_SIGNAL

        self.fingerprint = ConvFingerprint(self.n_signal, 
                                           fingerprint_size,
                                           lookback,
                                           fingerprint_dropout)
        self.readout_proj = nn.Sequential(
            nn.Dropout(readout_dropout),
            nn.Linear(enc_h_size, readout_size),
            nn.SELU(),
        )
        self.focus = nn.Linear(fingerprint_size, attn_window * n_heads)

    def forward(self, enc_input: torch.Tensor, 
                      enc_states: torch.Tensor) -> torch.Tensor:
        B = enc_input.size(0)

        fprint = self.fingerprint(enc_input[:, :, :self.n_signal])              # enc_in_size -> fingerprint_size
        scores = self.focus(fprint).view(B, self.attn_window, self.n_heads)     # attn_window, n_heads
        weights = scores / (scores.sum(dim=1, keepdim=True) + 1e-8)             # normalize

        readout = self.readout_proj(enc_states)                                 # enc_h_size -> readout_size
        readout = readout.permute(0, 2, 1)                                      # lookback, <-> readout_size
            
        heads = []
        for head_i in range(self.n_heads):
            w = weights[:, :, head_i]                                           # [B, attn_window]
            ro = readout.reshape(1, B * self.readout_size, self.lookback)       # [1, readout_size*B, lookback]
            kernel = w.repeat_interleave(self.readout_size, dim=0)              # [B*readout_size, attn_window]
            kernel = kernel.unsqueeze(1)                                        # [B*readout_size, 1, attn_window]
            out = F.conv1d(ro, kernel, groups=B * self.readout_size)            # [1, B*readout_size, horizon]
            heads.append(out.view(B, self.readout_size, self.horizon))          # [B, readout_size, horizon]

        attn = torch.stack(heads, dim=-1)                                       # [B, readout_size, horizon, n_heads]
        attn = attn.permute(0, 2, 1, 3).contiguous()                            # [B, horizon, readout_size, n_heads]
        return attn.view(B, self.horizon, self.readout_size * self.n_heads)     # [B, horizon, readout_size * n_heads]