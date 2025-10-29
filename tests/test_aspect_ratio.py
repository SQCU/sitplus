# python -m tests.test_aspect_ratio
import torch
from pytorch_wavelets import DWTForward, DWTInverse
from sitplus.generators.parametric import render_text_image
from sitplus.utils.dwt_tokenizer import coeffs_to_raw_tokens, raw_tokens_to_coeffs

def test_non_square_tokenizer():
    print("--- Running Aspect Ratio Tokenizer Test ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Create a non-square image (512x256)
    img = render_text_image("Wide Aspect", size=(512, 256)).to(device)
    print(f"Input image shape: {img.shape}")
    
    # 2. DWT
    dwt = DWTForward(J=4, wave='haar', mode='zero').to(device)
    idwt = DWTInverse(wave='haar', mode='zero').to(device)
    yl, yh = dwt(img)
    
    print(f"Coarsest approximation shape (yl): {yl.shape}") # Expect 32x16
    
    # 3. Tokenize
    grid_size = 8
    tokens_list, shapes_list = coeffs_to_raw_tokens(yl, yh, grid_size=grid_size)

    # 4. Verify shapes
    print(f"\nTarget tokens per scale: {grid_size * grid_size}")
    for i, t in enumerate(tokens_list):
        print(f"Scale {i} tokens: {t.shape}")
        assert t.shape[1] == grid_size * grid_size, f"Scale {i} must have {grid_size**2} tokens"
        
    # 5. Detokenize & Reconstruct
    yl_rec, yh_rec = raw_tokens_to_coeffs(tokens_list, shapes_list, grid_size=grid_size)
    img_rec = idwt((yl_rec, yh_rec))
    # Crop reconstructed image to match original size, as IDWT might pad
    img_rec = img_rec[:, :, :img.shape[2], :img.shape[3]]
    
    is_perfect = torch.allclose(img, img_rec, atol=1e-6)
    print(f"\nPerfect reconstruction? -> {is_perfect}")
    
    assert is_perfect, "Tokenizer failed on non-square image."
    print("\n--- ✅ Aspect Ratio Test Passed ---")

if __name__ == "__main__":
    test_non_square_tokenizer()