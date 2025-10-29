import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import create_block_mask

# Import the context manager for our test
from torch.backends.cuda import sdp_kernel 

from sitplus.utils.attention import RotaryPositionalEmbedding, apply_rotary_pos_emb
from sitplus.utils.masks import create_sit_mask_function

class RMSNorm(nn.Module):
    """
    A non-affine RMSNorm implementation, as used in many modern Transformers.
    No learnable weight or bias.
    """
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # The gain is now a simple scalar, not a learnable vector.
        self.scale = d_model ** 0.5

    def forward(self, x):
        # The formula is: (x / sqrt(mean(x^2) + eps)) * scale
        # For RMSNorm, the scale is just sqrt(dim), but many implementations omit it.
        # We'll use the standard (x * rsqrt(var + eps)) form.
        normed_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed_x

class SitEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_ff: int = 2048,
        n_heads: int = 8,
        rope_base: int = 500,
        max_seq_len: int = 4096,
        is_global_layer: bool = False,
        num_scales: int = 5,
        tokens_per_scale: int = 64,
        window_radius: int = 2, # Add this
        norm_type: str = "rmsnorm",
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.is_global_layer = is_global_layer
        self.tokens_per_scale = tokens_per_scale

        # --- Create two sets of all parameters: one for Coarse, one for Fine ---
        self._create_parameters("coarse", d_model, d_ff, use_qk_norm, norm_type)
        self._create_parameters("fine", d_model, d_ff, use_qk_norm, norm_type)
        
        # --- Shared Mechanisms ---
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len, base=rope_base)
        self.mask_fn = create_sit_mask_function(
            is_global_layer, num_scales, tokens_per_scale, window_radius
        )

    def _create_parameters(self, prefix, d_model, d_ff, use_qk_norm, norm_type):
        """Helper to avoid repetitive parameter creation."""
        NormLayer = RMSNorm if norm_type == "rmsnorm" else lambda d: nn.LayerNorm(d, elementwise_affine=False)
        self.add_module(f"input_norm_{prefix}", NormLayer(d_model))
        
        QKNORM = RMSNorm if norm_type == "rmsnorm" else lambda d: nn.LayerNorm(d, elementwise_affine=False)
        self.add_module(f"q_norm_{prefix}", QKNORM(self.head_dim) if use_qk_norm else nn.Identity())
        self.add_module(f"k_norm_{prefix}", QKNORM(self.head_dim) if use_qk_norm else nn.Identity())

        self.add_module(f"qkv_proj_{prefix}", nn.Linear(d_model, d_model * 3, bias=False))
        self.add_module(f"ffn_up_proj_{prefix}", nn.Linear(d_model, d_ff * 2, bias=True))
        self.add_module(f"out_proj_{prefix}", nn.Linear(d_model, d_model, bias=True))
        self.add_module(f"ffn_down_proj_{prefix}", nn.Linear(d_ff, d_model, bias=True))

    def _create_mask_fn(self):
        """Create the mask function for this layer."""
        tokens_per_scale = self.tokens_per_scale
        
        if self.is_global_layer:
            # Global: scale-causal only
            # pure tensor operations so pytorch can compile!
            def mask_fn(b, h, q_idx, kv_idx):
                q_scale = q_idx // tokens_per_scale
                kv_scale = kv_idx // tokens_per_scale
                return kv_scale <= q_scale
        else:
            # SWA: scale-causal + within-scale window
            # pure tensor operations so pytorch can compile!
            def mask_fn(b, h, q_idx, kv_idx):
                q_scale = q_idx // tokens_per_scale
                kv_scale = kv_idx // tokens_per_scale
                
                # SWA: within-scale window ONLY.
                # A token can ONLY see other tokens in its own scale.
                return q_scale == kv_scale
        
        return mask_fn

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x_coarse_in, x_fine_in = torch.split(x, [self.tokens_per_scale, T - self.tokens_per_scale], dim=1)
        
        # --- ACT I: PREPARE - Apply separate norms and projections ---

        norm_coarse = self.input_norm_coarse(x_coarse_in)
        norm_fine = self.input_norm_fine(x_fine_in)
        
        q_proj_c, k_proj_c, v_proj_c = self.qkv_proj_coarse(norm_coarse).chunk(3, dim=-1)
        q_proj_f, k_proj_f, v_proj_f = self.qkv_proj_fine(norm_fine).chunk(3, dim=-1)
        
        # Re-concatenate Q, K, V projections for a single attention call
        q_proj = torch.cat([q_proj_c, q_proj_f], dim=1)
        k_proj = torch.cat([k_proj_c, k_proj_f], dim=1)
        v_proj = torch.cat([v_proj_c, v_proj_f], dim=1)

        # --- ACT II: ATTEND - Unified attention over the full sequence ---
        
        q = q_proj.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_proj.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v_proj.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_pos_emb(q, k, self.rope)
        
        # Apply separate QK norms
        q_coarse, q_fine = torch.split(q, [self.tokens_per_scale, T - self.tokens_per_scale], dim=2)
        k_coarse, k_fine = torch.split(k, [self.tokens_per_scale, T - self.tokens_per_scale], dim=2)
        q = torch.cat([self.q_norm_coarse(q_coarse), self.q_norm_fine(q_fine)], dim=2)
        k = torch.cat([self.k_norm_coarse(k_coarse), self.k_norm_fine(k_fine)], dim=2)
        
        # Create block mask
        mask_fn = self._create_mask_fn()
        block_mask = create_block_mask(
            mask_fn,
            B=B,
            H=self.n_heads,
            Q_LEN=T,
            KV_LEN=T,
            _compile=True
        )
        # Attention is computed over the full sequence, with masks handling causality
        attn_output_full = flex_attention(q, k, v, block_mask=block_mask)
        attn_output_full = attn_output_full.transpose(1, 2).reshape(B, T, C)

        # --- ACT III: FINALIZE - Apply separate output projections and FFNs ---
        
        # Split attention output
        attn_out_c, attn_out_f = torch.split(attn_output_full, [self.tokens_per_scale, T - self.tokens_per_scale], dim=1)
        
        # Final Attention Path Output
        attn_final_c = self.out_proj_coarse(attn_out_c)
        attn_final_f = self.out_proj_fine(attn_out_f)
        
        # FFN Path (using the pre-attention norms)
        ffn_gate_c, ffn_up_c = self.ffn_up_proj_coarse(norm_coarse).chunk(2, dim=-1)
        ffn_out_c = self.ffn_down_proj_coarse(torch.nn.functional.silu(ffn_gate_c) * ffn_up_c)
        
        ffn_gate_f, ffn_up_f = self.ffn_up_proj_fine(norm_fine).chunk(2, dim=-1)
        ffn_out_f = self.ffn_down_proj_fine(torch.nn.functional.silu(ffn_gate_f) * ffn_up_f)
        
        # --- Final Residual Combination ---
        
        # Add coarse results to coarse residual
        output_coarse = x_coarse_in + attn_final_c + ffn_out_c
        # Add fine results to fine residual
        output_fine = x_fine_in + attn_final_f + ffn_out_f
        
        return torch.cat([output_coarse, output_fine], dim=1)


class SitEncoder(nn.Module):
    def __init__(
        self,
        raw_token_dims: list[int],
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        rope_base: int = 500,
        max_seq_len: int = 4096,
        num_scales: int = 5,
        tokens_per_scale: int = 64,
        window_radius: int = 2,
        norm_type: str = "rmsnorm",
        use_qk_norm: bool = True,
    ):
        super().__init__()
        
        # 1. Input Projection Layers
        # A unique projection for each scale is necessary because their
        # raw feature dimensions are different. This is the core design.
        self.projections = nn.ModuleList(
            [nn.Linear(dim, d_model) for dim in raw_token_dims]
        )
        
        # 2. Stack of Encoder Layers
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            # Every 3rd layer (0-indexed: 2, 5, 8...) is global
            is_global = (i + 1) % 3 == 0
            self.layers.append(
                SitEncoderLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    rope_base=rope_base,
                    max_seq_len=max_seq_len,
                    is_global_layer=is_global,
                    num_scales=num_scales,
                    tokens_per_scale=tokens_per_scale,
                    norm_type=norm_type,
                    use_qk_norm=use_qk_norm,
                    window_radius=window_radius, # Add this
                )
            )
            
        # 3. Final Normalization
        if norm_type == "rmsnorm":
            self.norm_out = RMSNorm(d_model)
        elif norm_type == "layernorm":
            self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, tokens_list: list[torch.Tensor]) -> torch.Tensor:
        # Apply the unique projection for each corresponding scale
        projected_tokens = [
            self.projections[i](tokens) for i, tokens in enumerate(tokens_list)
        ]
            
        # Concatenate into a single sequence for the Transformer
        x = torch.cat(projected_tokens, dim=1)
        
        # Pass through the stack of layers
        for layer in self.layers:
            x = layer(x)
            
        # Apply final normalization
        x = self.norm_out(x)
        
        return x