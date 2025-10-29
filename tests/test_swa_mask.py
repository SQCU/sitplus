# python -m tests.test_swa_mask 
import torch
from sitplus.utils.masks import create_sit_mask_function

def test_spatial_swa_mask():
    print("--- Running Spatial SWA Mask Test ---")
    
    # Config
    tokens_per_scale = 64 # 8x8 grid
    scale_edge_len = 8
    window_radius = 2

    # Get the mask function for an SWA layer
    mask_fn = create_sit_mask_function(
        is_global_layer=False,
        num_scales=5,
        tokens_per_scale=tokens_per_scale,
        window_radius=window_radius
    )
    # The returned function expects dummy b, h args
    test_mask = lambda q, k: mask_fn(0, 0, q, k)

    # --- Test Cases ---

    # Case 1: Center token attending to itself (should be True)
    center_idx = 10 # (y=1, x=2) in scale 0
    assert test_mask(center_idx, center_idx), "Token should attend to itself."
    print("✅ Token attends to self.")

    # Case 2: Center token attending to a close neighbor (should be True)
    neighbor_idx = 11 # (y=1, x=3), distance is (0, 1) -> within radius 2
    assert test_mask(center_idx, neighbor_idx), "Token should attend to close neighbor."
    print("✅ Token attends to close neighbor.")

    # Case 3: Center token attending to a corner of its window (should be True)
    corner_idx = 28 # (y=3, x=4), distance is (2, 2) -> at edge of radius 2
    assert test_mask(center_idx, corner_idx), "Token should attend to window corner."
    print("✅ Token attends to window corner.")
    
    # Case 4: Center token attending to a far token in the SAME scale (should be False)
    far_idx = 60 # (y=7, x=4), distance is (6, 2) -> outside radius 2
    assert not test_mask(center_idx, far_idx), "Token should NOT attend to far neighbor in same scale."
    print("✅ Token correctly blocked from far neighbor.")

    # Case 5: Center token attending to a token in a DIFFERENT scale (should be False)
    other_scale_idx = 70 # Scale 1
    assert not test_mask(center_idx, other_scale_idx), "Token should NOT attend to token in another scale."
    print("✅ Token correctly blocked from other scales.")

    print("\n--- ✅ Spatial SWA Mask Test Passed ---")

if __name__ == "__main__":
    test_spatial_swa_mask()