import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ══════════════════════════════════════════════════════════════════════════════
# RoPE encodes position via rotation matrices in 2D subspaces of the head
# dimension.  Advantages over the sinusoidal absolute PE used previously:
#   1. Captures *relative* distance naturally (dot product only depends on
#      the difference between positions, not absolute values).
#   2. Generalises to sequence lengths longer than those seen during training.
#   3. Fully compatible with Flash Attention / SDPA.
#
# Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position
# Embedding", 2021.  arXiv:2104.09864
# ══════════════════════════════════════════════════════════════════════════════

class RotaryPositionalEncoding(nn.Module):
    """Compute and cache RoPE frequency tensors for a given max length."""

    def __init__(self, d_model: int, num_heads: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        self.d_head = d_model // num_heads
        # Inverse frequency vector: θ_i = base^(-2i/d_head) for i in [0, d_head/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        self.register_buffer("inv_freq", inv_freq)  # [d_head // 2]
        # Build the initial cache
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        """Pre-compute sin/cos tables up to *seq_len* positions."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)          # [seq_len, d_head // 2]
        emb = torch.cat([freqs, freqs], dim=-1)         # [seq_len, d_head]
        # Register as buffers so they travel with .to(device) and .half()
        cos_cached = emb.cos()[None, None, :, :]         # [1, 1, seq_len, d_head]
        sin_cached = emb.sin()[None, None, :, :]         # [1, 1, seq_len, d_head]
        # Use persistent=False so these don't pollute state_dict size
        if hasattr(self, "cos_cached"):
            del self.cos_cached, self.sin_cached
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self._cached_len = seq_len

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (cos, sin) sliced to the requested sequence length.
        Automatically extends the cache if needed.
        """
        if seq_len > self._cached_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by splitting into two halves and swapping."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,   # [B, H, S, D]
    k: torch.Tensor,   # [B, H, S, D]
    cos: torch.Tensor,  # [1, 1, S, D]
    sin: torch.Tensor,  # [1, 1, S, D]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors."""
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ══════════════════════════════════════════════════════════════════════════════
# Legacy Sinusoidal Positional Encoding (kept for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════
# This class is retained so that old checkpoints trained with additive
# sinusoidal PE can still be loaded without errors.  New training runs
# should use RoPE via the `use_rope=True` flag on the Encoder / Decoder.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create position matrix using Sine/Cosine formula
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Fix: Change shape to [1, max_len, d_model] to match the batch-first format
        # of x which is [batch_size, seq_len, d_model] without needing a transpose
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        # Fix: Slice along dim=1 (seq_len) directly, no transpose needed anymore
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Head Attention (supports both legacy sinusoidal PE and RoPE)
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, q, k, v, mask=None, is_causal=False, rope_cos=None, rope_sin=None):
        """
        Parameters
        ----------
        q, k, v : Tensor [B, S, D]
        mask    : Optional additive attention mask.
        is_causal : If True, use causal masking (mutually exclusive with mask).
        rope_cos, rope_sin : Optional RoPE cosine/sine tensors [1, 1, S, D_head].
                             When provided, RoPE is applied to Q and K after projection.
        """
        batch_size = q.size(0)
        
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply Rotary Position Embedding when provided
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        
        # Pass the is_causal flag directly to PyTorch's optimized backend
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_output)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # Switched to GELU instead of ReLU for smoother gradient flow in deep models
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))