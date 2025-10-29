# python -m tests.test_tokenizer

import torch
from pytorch_wavelets import DWTForward, DWTInverse
from sitplus.generators.parametric import render_text_image
from sitplus.utils.dwt_tokenizer import coeffs_to_raw_tokens, raw_tokens_to_coeffs

def test_tokenizer_shapes_and_reversibility():
    print("--- Running Tokenizer Shape & Reversibility Test ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Data & DWT
    img = render_text_image("Tokens", size=(256, 256)).to(device)
    dwt = DWTForward(J=4, wave='haar', mode='zero').to(device)
    idwt = DWTInverse(wave='haar', mode='zero').to(device)
    
    yl, yh = dwt(img)
    
    # 2. Tokenize
    grid_size = 8
    tokens_list, shapes_list = coeffs_to_raw_tokens(yl, yh, grid_size=grid_size)
    
    # 3. Verify Shapes (The "Constant Tokens" Invariant)
    print(f"\nTarget tokens per scale: {grid_size * grid_size} ({grid_size}x{grid_size} grid)")
    
    # Scale 0 (Approx)
    print(f"Scale 0 (Approx) tokens: {tokens_list[0].shape}")
    assert tokens_list[0].shape[1] == 64, "Approx scale must have 64 tokens"
    
    # Detail scales (Coarse -> Fine)
    for i, t in enumerate(tokens_list[1:]):
        print(f"Scale {i+1} (Detail) tokens: {t.shape}")
        assert t.shape[1] == 64, f"Detail scale {i+1} must have 64 tokens"

    # 4. Detokenize & Reconstruct
    yl_rec, yh_rec = raw_tokens_to_coeffs(tokens_list, shapes_list, grid_size=grid_size)
    img_rec = idwt((yl_rec, yh_rec))
    
    # 5. Verify Perfect Reconstruction
    is_perfect = torch.allclose(img, img_rec, atol=1e-6)
    mse = torch.nn.functional.mse_loss(img, img_rec).item()
    
    print(f"\nReconstruction MSE: {mse:.2e}")
    print(f"Perfect reconstruction? -> {is_perfect}")
    
    assert is_perfect, "Tokenizer -> Detokenizer cycle failed to reconstruct perfectly."
    print("\n--- ✅ Tokenizer Test Passed ---")

if __name__ == "__main__":
    test_tokenizer_shapes_and_reversibility()