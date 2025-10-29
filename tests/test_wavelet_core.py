# python -m tests.test_wavelet_core
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

# Import our parametric image generator
from sitplus.generators.parametric import render_text_image

def test_dwt_reconstruction():
    """
    Checklist 1.1: Verify DWT/IDWT works and is perfectly invertible.
    This test also implicitly checks for CUDA compatibility by running on a GPU if available.
    """
    print("--- Running Wavelet Core Test ---")

    # 1. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data
    image_size = (256, 256)
    levels = 4 # Number of wavelet decomposition levels
    wavelet_type = 'haar'
    input_tensor = render_text_image(text="Recon", size=image_size).to(device)
    B, C, H, W = input_tensor.shape
    print(f"Input tensor shape: {input_tensor.shape}")

    # 3. Initialize Wavelet Transforms
    # J is the number of levels
    # mode='zero' is a padding mode, important for perfect reconstruction
    dwt = DWTForward(J=levels, wave=wavelet_type, mode='zero').to(device)
    idwt = DWTInverse(wave=wavelet_type, mode='zero').to(device)

    print(f"Initialized DWT with {levels} levels of '{wavelet_type}' wavelets.")

    # 4. Perform Forward and Inverse Transform
    try:
        # The forward transform returns low-pass (approximation) and high-pass (details) coefficients
        # yl: low-pass (B, C, H_low, W_low)
        # yh: list of high-pass bands for each level
        yl, yh = dwt(input_tensor)
        
        # The inverse transform takes these coefficients to reconstruct the image
        recon_tensor = idwt((yl, yh))

    except Exception as e:
        print(f"\n!!! AN ERROR OCCURRED DURING DWT/IDWT !!!")
        print(f"Error: {e}")
        print("This could be a problem with the CUDA installation or the library itself.")
        assert False, "Wavelet transform failed."

    # 5. Verify and Report
    print(f"Low-pass (yl) shape: {yl.shape}")
    print(f"High-pass levels returned: {len(yh)}")
    for i, band in enumerate(yh):
        print(f"  Level {i} high-pass (yh[{i}]) shape: {band.shape}")
    
    print(f"Reconstructed tensor shape: {recon_tensor.shape}")

    # The reconstructed tensor might have slightly different dimensions due to padding/pooling
    # We must crop it back to the original size for a fair comparison.
    recon_tensor_cropped = recon_tensor[:, :, :H, :W]

    # Calculate reconstruction error
    mse = F.mse_loss(input_tensor, recon_tensor_cropped).item()
    is_perfect_recon = torch.allclose(input_tensor, recon_tensor_cropped, atol=1e-6)

    print(f"\n--- Verification ---")
    print(f"Mean Squared Error (MSE): {mse:.2e}")
    print(f"Is reconstruction perfect (torch.allclose)? -> {is_perfect_recon}")

    # Save images for visual inspection
    from torchvision.utils import save_image
    save_image(input_tensor, "test_wavelet_original.png")
    save_image(recon_tensor_cropped, "test_wavelet_reconstructed.png")
    print("Saved 'test_wavelet_original.png' and 'test_wavelet_reconstructed.png' for visual check.")
    
    assert is_perfect_recon, f"Reconstruction failed! MSE is too high: {mse}"
    print("\n--- ✅ Test Passed ---")


if __name__ == "__main__":
    test_dwt_reconstruction()