# sitplus.utils.attention
import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    """
    A 1D Rotary Positional Embedding (RoPE) module.
    This is the simplest, most effective positional encoding for Transformers.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        """
        Initializes the RoPE module.

        Args:
            dim (int): The dimension of the head embeddings.
            max_seq_len (int): The maximum sequence length to pre-compute for.
            base (int): The base for the rotary frequencies.
                
                IMPORTANT LITERATURE NOTE (arXiv:2310.05209):
                The default base of 10000 is often suboptimal for extrapolation.
                - For robust extrapolation to unpredictable lengths, a SMALLER base
                  (e.g., 500) fine-tuned on a longer context (e.g., 16k) performs best.
                - For extrapolation to a KNOWN fixed length (e.g., 100k), a LARGER
                  base (e.g., 1,000,000) can be calculated and used.
                
                This implementation defaults to 10000 for compatibility but
                using a different, empirically-justified base is strongly recommended.
        """
        super().__init__()
        
        # Create the inverse frequency buffer
        # Shape: (dim / 2)
        # The only change is here: 10000 is replaced by the `base` parameter.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Create the sinusoidal cache
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # freqs shape is (max_seq_len, dim / 2)

        
        # Store as (1, 1, max_seq_len, dim / 2) for broadcasting
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :])
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor):
        """
        Apply RoPE to a tensor.
        Args:
            x: Input tensor of shape (B, H, T, D) or (B, T, D)
               B=batch, H=heads, T=sequence_len, D=head_dim
        """
        # Get the sequence length from the input tensor
        seq_len = x.shape[-2]
        
        # Get pre-computed cos and sin values
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]

        # Reshape x for rotation: (..., T, D/2, 2)
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x_reshaped.unbind(dim=-1)

        # Apply the rotation
        # rotated_x1 = x1 * cos - x2 * sin
        # rotated_x2 = x1 * sin + x2 * cos
        # This is equivalent to complex number multiplication:
        # (x1 + i*x2) * (cos + i*sin)
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Reassemble the rotated tensor
        rotated_x = torch.stack((rotated_x1, rotated_x2), dim=-1).reshape(x.shape)
        
        return rotated_x.type_as(x)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, rope: RotaryPositionalEmbedding):
    """
    Helper function to apply RoPE to query and key tensors.
    """
    q_embed = rope(q)
    k_embed = rope(k)
    return q_embed, k_embed