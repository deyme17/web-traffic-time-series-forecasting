from abc import ABC, abstractmethod
from utils import Registry
import torch.nn.functional as F
import torch.nn as nn
import torch

ATTENTIONS = Registry()



class Attention(nn.Module, ABC):
    """Base attention mechanism implementation class."""
    def __init__(self, 
                 query_size: int, 
                 key_size: int, 
                 attn_size: int = 128):
        super().__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.attn_size = attn_size
        self.attn_weights = None
    
    @abstractmethod
    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Calculate and save attention weights. Return attention context."""
        pass



@ATTENTIONS.register("additive")
class AdditiveAttention(Attention):
    """Bahdanau-style additive attention."""
    def __init__(self, 
                 query_size: int, 
                 key_size: int, 
                 attn_size: int = 128):
        super().__init__(query_size, key_size, attn_size)
        self.v = nn.Linear(self.attn_size, 1, bias=False)
        self.W_q= nn.Linear(self.query_size, self.attn_size, bias=False)
        self.W_k = nn.Linear(self.key_size, self.attn_size, bias=False)

    def forward(self, query, keys):
        scores = self.v(
            F.tanh(self.W_q(query).unsqueeze(1) + self.W_k(keys))
        ).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        self.attn_weights = weights.detach()
        context = (weights.unsqueeze(-1) * keys).sum(dim=1)
        return context
        


@ATTENTIONS.register("dot")
class DotProductAttention(Attention):
    """Scaled dot-product attention (transformer-style)."""
    def __init__(self, 
                 query_size: int, 
                 key_size: int, 
                 attn_size: int = 128):
        super().__init__(query_size, key_size, attn_size)
        self.W_q = nn.Linear(query_size, attn_size, bias=False)
        self.W_k = nn.Linear(key_size, attn_size, bias=False)
        self.scale = attn_size**0.5
    
    def forward(self, query, keys):
        Q = self.W_q(query).unsqueeze(1)
        K = self.W_k(keys)
        scores = (Q * K).sum(-1) / self.scale
        weights = F.softmax(scores, dim=-1)
        self.attn_weights = weights.detach()
        context = (weights.unsqueeze(-1) * keys).sum(dim=1)
        return context



@ATTENTIONS.register("multihead")
class MultiHeadAttention(Attention):
    """Lightweight multi-head attention over encoder states."""
    def __init__(self, 
                 query_size: int, 
                 key_size: int, 
                 attn_size: int = 128,
                 n_heads: int = 4):
        super().__init__(query_size, key_size, attn_size)
        assert attn_size % n_heads == 0, "attn_size must be divisible by n_heads"
        self.n_heads = n_heads
        self.proj_q = nn.Linear(query_size, attn_size, bias=False)
        self.proj_k = nn.Linear(key_size, attn_size, bias=False)
        self.proj_v = nn.Linear(key_size, attn_size, bias=False)
        self.mha = nn.MultiheadAttention(attn_size, n_heads, batch_first=True)
        self.out_proj = nn.Linear(attn_size, key_size, bias=False)
    
    def forward(self, query, keys):
        Q = self.proj_q(query).unsqueeze(1)
        K = self.proj_k(keys)
        V = self.proj_v(keys)
        context, weights = self.mha(Q, K, V)
        self.attn_weights = weights.detach().squeeze(1)
        context = self.out_proj(context.squeeze(1))
        return context