from .attentions import ATTENTIONS, Attention
import torch.nn as nn


def build_attention(attn_type: str, 
                    query_size: int, 
                    key_size: int, 
                    attn_size: int, 
                    n_heads: int = 4) -> Attention:
    """Factory for attention modules ("additive", "dot", "multihead", "none")."""
    if attn_type == "none":
        return None
    cls = ATTENTIONS.get(attn_type)
    if cls is None:
        raise ValueError(f"Unknown attn_type={attn_type}. Choose from {list(ATTENTIONS)} or 'none'.")
    kwargs = dict(query_size=query_size, key_size=key_size, attn_size=attn_size)
    if attn_type == "multihead":
        kwargs["n_heads"] = n_heads
    return cls(**kwargs)