import torch
import torch.nn as nn
from .attention import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding, RotaryPositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Pre-LayerNorm Architecture 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask, rope_cos=None, rope_sin=None):
        # Self-Attention with Pre-LN
        nx = self.norm1(x)
        attn_out = self.self_attn(nx, nx, nx, src_mask, rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + self.dropout(attn_out)
        
        # Feed Forward
        nx = self.norm2(x)
        ff_out = self.feed_forward(nx)
        x = x + self.dropout(ff_out)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float, use_rope: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.use_rope = use_rope

        if use_rope:
            self.rope = RotaryPositionalEncoding(d_model, num_heads)
            # Dropout applied after embedding (no additive PE)
            self.emb_dropout = nn.Dropout(dropout)
        else:
            # Legacy sinusoidal positional encoding for backward compatibility
            self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, src, src_mask):
        x = self.embedding(src) * (self.d_model ** 0.5)

        if self.use_rope:
            x = self.emb_dropout(x)
            seq_len = src.size(1)
            rope_cos, rope_sin = self.rope(seq_len)
        else:
            x = self.pos_encoding(x)
            rope_cos, rope_sin = None, None
        
        for layer in self.layers:
            x = layer(x, src_mask, rope_cos=rope_cos, rope_sin=rope_sin)
            
        return self.norm(x)