import torch
import torch.nn as nn
from .encoder import Encoder  
from .decoder import Decoder  

class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size: int, 
        tgt_vocab_size: int, 
        src_pad_idx: int, 
        tgt_pad_idx: int, 
        d_model: int = 256,   # Adjusted to match the value in train.py
        num_layers: int = 6, 
        num_heads: int = 8,   # Adjusted to align with d_model=256 (256/8=32 per head)
        # Fix: d_ff should be 4x of d_model (standard), not 16x
        # d_model=256 -> d_ff=1024 (matches train.py)
        d_ff: int = 1024, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        
        # Initialize Weights with Xavier/Glorot to help the model start learning better
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 1. Share Embedding between Encoder and Decoder
        self.encoder.embedding.weight = self.decoder.embedding.weight
        # 2. Share weights from Embedding to the Output Linear Layer
        self.decoder.fc_out.weight = self.decoder.embedding.weight

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        # Create Mask for the Source side (Encoder)
        # 0.0 means attend, -inf means skip this token (e.g., <pad>)
        mask = (src == self.src_pad_idx).unsqueeze(1).unsqueeze(2) # Shape: [N, 1, 1, src_len]
        return mask.masked_fill(mask, float('-inf')).float()

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        N, tgt_len = tgt.shape
        
        # 1. Pad Mask (hide <pad> tokens)
        pad_mask = (tgt == self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 2. Causal Mask (hide future tokens to prevent the Decoder from looking ahead at the answer)
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).type(torch.bool)
        causal_mask = ~causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Combine both masks
        mask = pad_mask | causal_mask
        return mask.float().masked_fill(mask, float('-inf'))

    def forward(self, src, tgt):
        # Create masks before feeding into the model
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # Pass through Encoder
        enc_output = self.encoder(src, src_mask)
        
        # Pass through Decoder using the Encoder output for assistance
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        return output