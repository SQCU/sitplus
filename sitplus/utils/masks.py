from functools import partial
import torch
from torch.nn.attention.flex_attention import _vmap_for_bhqkv

# --- Helper Functions ---
def create_scale_boundaries(num_scales: int, tokens_per_scale: int) -> list[tuple[int, int]]:
    """Creates a list of (start, end) index tuples for each scale."""
    boundaries = []
    for s in range(num_scales):
        start = s * tokens_per_scale
        end = (s + 1) * tokens_per_scale
        boundaries.append((start, end))
    return boundaries

def find_scale_from_idx(token_idx: int, tokens_per_scale: int) -> int:
    """Quickly finds the scale ID for a given token index."""
    return token_idx // tokens_per_scale

# --- Flex Attention Predicate Functions ---
def scale_causal_mask_fn(b: int, h: int, q_idx: int, kv_idx: int, tokens_per_scale: int) -> bool:
    """
    Predicate for scale-causality. A token can only attend to tokens
    in its own scale or previous scales.
    
    Args:
        b: batch index (unused, but required by flex_attention)
        h: head index (unused, but required by flex_attention)
        q_idx: query token index
        kv_idx: key/value token index
        tokens_per_scale: number of tokens per scale
    
    Returns:
        True if attention is ALLOWED.
    """
    q_scale = find_scale_from_idx(q_idx, tokens_per_scale)
    kv_scale = find_scale_from_idx(kv_idx, tokens_per_scale)
    return kv_scale <= q_scale

def sliding_window_mask_fn(b: int, h: int, q_idx: int, kv_idx: int, tokens_per_scale: int) -> bool:
    """
    Predicate for sliding window attention (SWA).
    Only allows attention between tokens within the same scale.
    
    Args:
        b: batch index (unused, but required by flex_attention)
        h: head index (unused, but required by flex_attention)
        q_idx: query token index
        kv_idx: key/value token index
        tokens_per_scale: number of tokens per scale
    
    Returns:
        True if attention is ALLOWED.
    """
    q_scale = find_scale_from_idx(q_idx, tokens_per_scale)
    kv_scale = find_scale_from_idx(kv_idx, tokens_per_scale)
    # This predicate's only job is to enforce locality WITHIN a scale.
    # The scale_causal_mask will handle the cross-scale restrictions.
    return q_scale == kv_scale

def compose_masks(*mask_fns):
    """
    Higher-order function to compose multiple mask predicates with a logical AND.
    
    All mask functions must have signature: (b, h, q_idx, kv_idx) -> bool
    """
    def combined_mask(b, h, q_idx, kv_idx):
        return all(fn(b, h, q_idx, kv_idx) for fn in mask_fns)
    return combined_mask

def spatial_sliding_window_mask_fn(
    q_idx: torch.Tensor, 
    kv_idx: torch.Tensor, 
    tokens_per_scale: int,
    window_radius: int
) -> torch.Tensor:
    """
    Predicate for a true 2D sliding window.
    Only allows attention between tokens that are spatially close *within the same scale*.
    
    Args:
        q_idx, kv_idx: Can be tensors of indices for torch.compile.
        tokens_per_scale: Total tokens in a scale (e.g., 64 for 8x8).
        window_radius: The Chebyshev distance (max coordinate difference) allowed.
    
    Returns:
        A boolean tensor indicating if attention is ALLOWED.
    """
    # 1. Check if tokens are in the same scale. If not, block attention.
    q_scale = q_idx // tokens_per_scale
    kv_scale = kv_idx // tokens_per_scale
    same_scale_mask = (q_scale == kv_scale)
    
    # If not in the same scale, we can immediately return False for those pairs.
    # We'll calculate spatial distance for all and then mask at the end.

    # 2. Convert 1D indices to 2D coordinates within their scale.
    scale_edge_len = int(tokens_per_scale**0.5)
    assert scale_edge_len * scale_edge_len == tokens_per_scale, "tokens_per_scale must be a perfect square."
    
    q_local_idx = q_idx % tokens_per_scale
    kv_local_idx = kv_idx % tokens_per_scale
    
    q_y, q_x = q_local_idx // scale_edge_len, q_local_idx % scale_edge_len
    kv_y, kv_x = kv_local_idx // scale_edge_len, kv_local_idx % scale_edge_len
    
    # 3. Calculate Chebyshev distance (max of coordinate-wise absolute differences)
    dist_y = torch.abs(q_y - kv_y)
    dist_x = torch.abs(q_x - kv_x)
    chebyshev_dist = torch.max(dist_y, dist_x)
    
    # 4. Create the window mask and combine with the same-scale mask
    window_mask = (chebyshev_dist <= window_radius)
    
    return same_scale_mask & window_mask


# --- Update the Mask Factory ---

def create_sit_mask_function(
    is_global_layer: bool, 
    num_scales: int, 
    tokens_per_scale: int,
    window_radius: int = 2 # Add the new parameter with a default
):
    """
    Factory that returns a fully-tensorized, JIT-compatible predicate function.
    """
    scale_edge_len = int(tokens_per_scale**0.5)
    
    def mask_mod(b, h, m, n):
        # m is a tensor of q_indices, n is a tensor of kv_indices.
        # This function is designed to be traced by torch.compile/vmap.
        
        q_scale = m // tokens_per_scale
        kv_scale = n // tokens_per_scale

        if is_global_layer:
            # Global Layer: scale-causal mask.
            # This is a pure tensor operation.
            return kv_scale <= q_scale
        else:
            # SWA Layer: spatial sliding window.
            
            # 1. Same-scale check (tensor op)
            same_scale_mask = (q_scale == kv_scale)
            
            # 2. Spatial window check (tensor ops)
            q_local_idx = m % tokens_per_scale
            kv_local_idx = n % tokens_per_scale
            
            q_y = q_local_idx // scale_edge_len
            q_x = q_local_idx % scale_edge_len
            kv_y = kv_local_idx // scale_edge_len
            kv_x = kv_local_idx % scale_edge_len
            
            dist_y = torch.abs(q_y - kv_y)
            dist_x = torch.abs(q_x - kv_x)
            
            # The `max` function on tensors is traceable.
            window_mask = torch.max(dist_y, dist_x) <= window_radius
            
            # 3. Combine masks (logical AND is a traceable tensor op)
            return same_scale_mask & window_mask

    return mask_mod

"""
flex_attention(
    query,           # (B, H, T, D)
    key,             # (B, H, T, D)
    value,           # (B, H, T, D)
    score_mod=None,  # Optional: (b, h, q_idx, kv_idx, score) -> modified_score
    block_mask=None  # Optional: BlockMask object
)

# When you pass a plain function (not BlockMask), flex_attention treats it as score_mod
# Score mod functions receive 5 args: (b, h, q_idx, kv_idx, score)
# But you want a MASK, not a score modifier!
"""