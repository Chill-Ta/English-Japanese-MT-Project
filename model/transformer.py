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
        d_model: int = 512,   # Adjusted to match the value in train.py
        num_layers: int = 6, 
        num_heads: int = 8,   # 512/8 = 64 dims per head
        # Fix: d_ff should be 4x of d_model (standard), not 16x
        d_ff: int = 2048, 
        dropout: float = 0.1,
        use_rope: bool = False
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, use_rope=use_rope)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, use_rope=use_rope)
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
        """
        Build a combined causal + pad mask for the decoder self-attention.
        Returns shape [N, 1, tgt_len, tgt_len] with 0.0 = attend, -inf = block.
        
        This combined approach is required because PyTorch's
        scaled_dot_product_attention forbids passing both attn_mask AND
        is_causal=True simultaneously.
        """
        tgt_len = tgt.size(1)
        
        # 1. Causal mask: lower-triangular (attend to past + current, block future)
        #    Shape: [tgt_len, tgt_len]
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1
        ).bool()  # True = BLOCK
        
        # 2. Pad mask: block attention to/from <pad> positions
        #    Shape: [N, 1, 1, tgt_len] -> broadcasts over query positions
        pad_mask = (tgt == self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)  # True = BLOCK
        
        # 3. Combine: block if EITHER causal or pad says block
        #    causal_mask broadcasts from [tgt_len, tgt_len] to [N, 1, tgt_len, tgt_len]
        #    pad_mask broadcasts from [N, 1, 1, tgt_len] to [N, 1, tgt_len, tgt_len]
        combined = causal_mask.unsqueeze(0).unsqueeze(0) | pad_mask
        
        # 4. Convert to additive float mask: 0.0 = attend, -inf = block
        return combined.float().masked_fill(combined, float('-inf'))

    def forward(self, src, tgt):
        # Create masks before feeding into the model
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # Pass through Encoder
        enc_output = self.encoder(src, src_mask)
        
        # Pass through Decoder using the Encoder output for assistance
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        return output