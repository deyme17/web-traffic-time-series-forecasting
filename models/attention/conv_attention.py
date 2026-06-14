import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvFingerprint(nn.Module):
    """CNN that produces a 'fingerprint' of the input timeseries."""
    def __init__(self, in_ch: int = 1, out_size: int = 16, lookback: int = 365):
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
            nn.Linear(64 * (lookback // 8), 512), nn.SiLU(),
            nn.Linear(512, out_size), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) # [B, T, <-> C]
        x = self.convnet(x)
        x = x.flatten(1)
        return self.fc(x)



class ConvAttention(nn.Module):
    """Attention on the encoder fingerprint given by depthwise conv."""
    def __init__(self,
                 enc_in_size: int,
                 enc_h_size: int,
                 readout_size: int,
                 fingerprint_size: int,
                 attn_window: int,
                 horizon: int,
                 lookback: int,
                 n_heads: int = 4):
        """
        Args:
            enc_in_size: Encoder input size.
            enc_h_size: Encoder hidden size.
            readout_size: Compressed readout depth.
            fingerprint_size: CNN output size.
            attn_window: Lookback - horizon + 1.
            horizon: Prediction window.
            lookback: Past timestep size.
            n_heads: Number of attention heads.
        """
        super().__init__()
        self.n_heads = n_heads
        self.readout_size = readout_size
        self.horizon = horizon
        self.lookback = lookback
        self.attn_window = attn_window

        self.fingerprint = ConvFingerprint(enc_in_size, 
                                           fingerprint_size,
                                           lookback)
        self.readout_proj = nn.Sequential(
            nn.Linear(enc_h_size, readout_size),
            nn.SiLU(),
        )
        self.focus = nn.Linear(fingerprint_size, attn_window * n_heads)

    def forward(self, enc_input: torch.Tensor, 
                      enc_states: torch.Tensor) -> torch.Tensor:
        B = enc_input.size(0)

        fprint = self.fingerprint(enc_input)                                    # enc_in_size -> fingerprint_size
        scores = self.focus(fprint)                                             # attn_window * n_heads
        scores = scores.view(B, self.attn_window, self.n_heads)                 # attn_window, n_heads
        weights = F.softmax(scores, dim=1)                                      # normalize
        weights = weights.unsqueeze(1).unsqueeze(1)                             # [B, 1, 1, ...]

        readout = self.readout_proj(enc_states)                                 # enc_h_size -> readout_size
        readout = readout.permute(0, 2, 1)                                      # lookback, <-> readout_size
        readout = readout.unfold(dimension=2, size=self.attn_window, step=1)    # [B, readout_size, horizon, attn_window]
        readout = readout.unsqueeze(-1)                                         # [B, readout_size, horizon, attn_window, 1]

        attention = (readout * weights).sum(dim=3)                              # [B, readout_size, horizon, n_heads]
        attention = attention.permute(0, 2, 1, 3).contiguous()                  # [B, horizon, readout_size, n_heads]
        return attention.view(B, self.horizon, self.readout_size*self.n_heads)  # [B, horizon, readout_size * n_heads]