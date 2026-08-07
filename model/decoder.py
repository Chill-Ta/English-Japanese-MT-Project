import torch
import torch.nn as nn
from .attention import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding, RotaryPositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # Update the forward function in DecoderLayer class
    def forward(self, x, enc_output, src_mask, tgt_mask, rope_cos=None, rope_sin=None):
        # 1. Masked Self-Attention (tgt_mask contains combined causal + pad mask)
        nx = self.norm1(x)
        # is_causal=False because tgt_mask already includes the causal mask
        attn_out = self.self_attn(nx, nx, nx, mask=tgt_mask, is_causal=False,
                                  rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + self.dropout(attn_out)
        
        # 2. Cross-Attention (is_causal is False by default here)
        # Note: RoPE is NOT applied to cross-attention — the encoder already
        # encoded positional information into its output representations.
        nx = self.norm2(x)
        attn_out = self.cross_attn(nx, enc_output, enc_output, mask=src_mask)
        x = x + self.dropout(attn_out)
        
        # 3. Feed Forward
        nx = self.norm3(x)
        ff_out = self.feed_forward(nx)
        x = x + self.dropout(ff_out)
        
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float, use_rope: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.use_rope = use_rope

        if use_rope:
            self.rope = RotaryPositionalEncoding(d_model, num_heads)
            self.emb_dropout = nn.Dropout(dropout)
        else:
            # Legacy sinusoidal positional encoding
            self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.embedding(tgt) * (self.d_model ** 0.5)

        if self.use_rope:
            x = self.emb_dropout(x)
            seq_len = tgt.size(1)
            rope_cos, rope_sin = self.rope(seq_len)
        else:
            x = self.pos_encoding(x)
            rope_cos, rope_sin = None, None
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask,
                      rope_cos=rope_cos, rope_sin=rope_sin)
            
        x = self.norm(x)
        return self.fc_out(x)