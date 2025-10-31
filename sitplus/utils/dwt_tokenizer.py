# sitplus.utils.dwt_tokenizer

import torch

def patchify_tensor(x: torch.Tensor, grid_size_h: int = 8, grid_size_w: int = 8) -> torch.Tensor:
    """
    Reshapes a (B, C, H, W) tensor into (B, grid_h*grid_w, flattened_patch).
    Handles non-square inputs.
    """
    B, C, H, W = x.shape
    assert H % grid_size_h == 0, f"Height {H} must be divisible by grid height {grid_size_h}"
    assert W % grid_size_w == 0, f"Width {W} must be divisible by grid width {grid_size_w}"
    
    patch_size_h = H // grid_size_h
    patch_size_w = W // grid_size_w
    
    x_reshaped = x.reshape(B, C, grid_size_h, patch_size_h, grid_size_w, patch_size_w)
    x_permuted = x_reshaped.permute(0, 2, 4, 1, 3, 5)
    x_patched = x_permuted.reshape(B, grid_size_h * grid_size_w, -1)
    
    return x_patched, (C, H, W)

def unpatchify_tensor(x_patched: torch.Tensor, original_shape: tuple[int, int, int], grid_size_h: int = 8, grid_size_w: int = 8) -> torch.Tensor:
    """Reverses patchify_tensor for non-square inputs."""
    C, H, W = original_shape
    patch_size_h = H // grid_size_h
    patch_size_w = W // grid_size_w
    B = x_patched.shape[0]
    
    x_unflattened = x_patched.reshape(B, grid_size_h, grid_size_w, C, patch_size_h, patch_size_w)
    x_permuted = x_unflattened.permute(0, 3, 1, 4, 2, 5)
    x_reconstructed = x_permuted.reshape(B, C, H, W)
    
    return x_reconstructed

def coeffs_to_raw_tokens(yl: torch.Tensor, yh: list[torch.Tensor], grid_size: int = 8) -> tuple[torch.Tensor, list[tuple]]:
    """
    Converts DWT coefficients into a single flat sequence of raw (unprojected) tokens.
    
    Returns:
        full_sequence: (B, total_tokens, max_raw_dim) - Padded for now if dims match, 
                       OR we might just return list of tensors if we want to project later.
                       Actually, let's return a LIST of token tensors, one per scale.
                       This is better because their feature dimensions will wildly differ.
        shapes: List of original shapes for perfect reconstruction.
    """
    tokens_list = []
    shapes_list = []
    
    # 1. Process Approximation (yl)
    # yl shape: (B, C, H_n, W_n)
    yl_tokens, yl_shape = patchify_tensor(yl, grid_size)
    tokens_list.append(yl_tokens)
    shapes_list.append(yl_shape)
    
    # 2. Process Details (yh) from coarsest to finest
    # Pytorch-wavelets returns finest first (yh[0]), so we reverse to go coarse->fine
    # Actually, standard DWT order is usually coarse->fine for token sequences.
    # Let's stick to the order provided by yh for now, but be aware of it.
    # If yh[0] is 128x128 (finest) and yh[3] is 16x16 (coarsest).
    # A better sequence order might be: Approx, Detail_Coarsest, ..., Detail_Finest
    
    # Let's use reversing to get Coarse -> Fine detail order
    # yh_coarse_to_fine = reversed(yh) 
    
    for scale_coeffs in reversed(yh):
        # scale_coeffs shape: (B, C, 3, H_i, W_i)
        B, C, _, H_i, W_i = scale_coeffs.shape
        
        # Merge the 3 subbands into the channel dimension for patching
        # (B, C*3, H_i, W_i)
        coeffs_merged = scale_coeffs.reshape(B, C * 3, H_i, W_i)
        
        tokens, shape = patchify_tensor(coeffs_merged, grid_size)
        tokens_list.append(tokens)
        shapes_list.append((C, 3, H_i, W_i)) # Store original 5D shape info
        
    return tokens_list, shapes_list

def raw_tokens_to_coeffs(tokens_list: list[torch.Tensor], shapes_list: list[tuple], grid_size: int = 8):
    """
    Reverses coeffs_to_raw_tokens.
    """
    # 1. Reconstruct yl
    yl_shape = shapes_list[0]
    yl = unpatchify_tensor(tokens_list[0], yl_shape, grid_size)
    
    # 2. Reconstruct yh (remember we reversed them!)
    yh = []
    # Indices 1 to end are the details, ordered Coarse -> Fine.
    # We need to build the yh list back in original order (Finest -> Coarsest)
    
    detail_tokens = tokens_list[1:]
    detail_shapes = shapes_list[1:]
    
    for tokens, shape_info in zip(reversed(detail_tokens), reversed(detail_shapes)):
        C, bands, H_i, W_i = shape_info
        # Shape info for unpatchify needs to be (C*3, H_i, W_i)
        merged_shape = (C * bands, H_i, W_i)
        
        coeffs_merged = unpatchify_tensor(tokens, merged_shape, grid_size)
        
        # Reshape back to 5D (B, C, 3, H_i, W_i)
        B = coeffs_merged.shape[0]
        coeffs_original = coeffs_merged.reshape(B, C, bands, H_i, W_i)
        yh.append(coeffs_original)
        
    return yl, yh