# python -m tests.test_encoder_layer
import torch
from sitplus.utils.encoder import SitEncoderLayer


def test_encoder_and_masks():
    print("--- Running Encoder Layer and Masking Test ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # --- Config ---
    d_model = 256
    seq_len = 320
    batch_size = 1

    swa_layer = SitEncoderLayer(d_model=d_model, is_global_layer=False, max_seq_len=seq_len).to(device)
    global_layer = SitEncoderLayer(d_model=d_model, is_global_layer=True, max_seq_len=seq_len).to(device)
    
    # --- Canary Test 1: Fine-grained token should NOT affect coarse-grained token ---
    print("\n--- Testing: Fine scale cannot influence coarse scale ---")
    coarse_target_idx = 10  # Scale 0
    fine_canary_idx = 70    # Scale 1

    canary_input = torch.zeros(batch_size, seq_len, d_model, device=device)
    canary_input[:, coarse_target_idx, :] = 1.0
    canary_input[:, fine_canary_idx, :] = 99.0
    
    base_input = torch.zeros(batch_size, seq_len, d_model, device=device)
    base_input[:, coarse_target_idx, :] = 1.0

    # For BOTH layer types, the canary should have no effect on the target
    swa_out_canary = swa_layer(canary_input)
    global_out_canary = global_layer(canary_input)
    
    swa_out_base = swa_layer(base_input)
    global_out_base = global_layer(base_input)

    assert torch.allclose(swa_out_base[:, coarse_target_idx], swa_out_canary[:, coarse_target_idx], atol=1e-5), \
        "SWA failed: Fine canary affected coarse target."
    print("✅ SWA layer correctly blocks fine -> coarse attention.")
    
    assert torch.allclose(global_out_base[:, coarse_target_idx], global_out_canary[:, coarse_target_idx], atol=1e-5), \
        "Global failed: Fine canary affected coarse target (violates scale-causality)."
    print("✅ Global layer correctly blocks fine -> coarse attention (scale-causal).")

    # --- Canary Test 2: Coarse-grained token SHOULD affect fine-grained token in GLOBAL layer ---
    print("\n--- Testing: Coarse scale CAN influence fine scale (in Global layer) ---")
    fine_target_idx = 70    # Scale 1
    coarse_canary_idx = 10  # Scale 0

    canary_input = torch.zeros(batch_size, seq_len, d_model, device=device)
    canary_input[:, fine_target_idx, :] = 1.0
    canary_input[:, coarse_canary_idx, :] = 99.0
    
    base_input = torch.zeros(batch_size, seq_len, d_model, device=device)
    base_input[:, fine_target_idx, :] = 1.0

    swa_out_canary = swa_layer(canary_input)
    global_out_canary = global_layer(canary_input)
    
    swa_out_base = swa_layer(base_input)
    global_out_base = global_layer(base_input)

    # For SWA, the canary should have no effect
    assert torch.allclose(swa_out_base[:, fine_target_idx], swa_out_canary[:, fine_target_idx], atol=1e-5), \
        "SWA failed: Coarse canary affected fine target."
    print("✅ SWA layer correctly blocks coarse -> fine attention.")

    # For GLOBAL, the canary SHOULD have an effect
    assert not torch.allclose(global_out_base[:, fine_target_idx], global_out_canary[:, fine_target_idx], atol=1e-5), \
        "Global failed: Coarse canary DID NOT affect fine target."
    print("✅ Global layer correctly allows coarse -> fine attention (scale-causal).")

    print("\n--- ✅ Encoder Layer and Masks Test Passed ---")

if __name__ == "__main__":
    test_encoder_and_masks()