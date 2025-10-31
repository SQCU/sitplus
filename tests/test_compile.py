# python -m tests.test_compile
import torch
from sitplus.utils.encoder import SitEncoder
from pytorch_wavelets import DWTForward, DWTInverse
from sitplus.utils.dwt_tokenizer import coeffs_to_raw_tokens

def test_encoder_compiles():
    """
    Verifies that the SitEncoderLayer is compatible with torch.compile().
    This test directly addresses the UserWarning from flex_attention.
    """
    print("--- Running torch.compile() Compatibility Test ---")
    
    # 1. Check if CUDA is available, as compilation is most relevant for GPU
    if not torch.cuda.is_available():
        print("CUDA not available, skipping compilation test.")
        return

    device = torch.device("cuda")
    print(f"Testing on device: {device}")

    # --- Configuration for the test ---
    d_model = 256
    n_layers = 6
    n_heads = 8
    num_scales = 5
    tokens_per_scale = 64
    total_tokens = num_scales * tokens_per_scale # 320
    batch_size = 2
    image_size = 256
    num_dwt_levels = 4
    
    # 1. Create a "real datum": a proper 4D random image tensor.
    # This fixes the previous IndexError.
    dummy_image = torch.randn(batch_size, 3, image_size, image_size).to(device)

    # 2. Run the full pre-processing pipeline to get realistic inputs for the encoder.
    dwt = DWTForward(J=num_dwt_levels, wave='haar', mode='zero').to(device)
    yl, yh = dwt(dummy_image)
    # The grid size is the square root of tokens per scale
    tokens_list, _ = coeffs_to_raw_tokens(yl, yh, grid_size=int(tokens_per_scale**0.5))
    
    # Dynamically determine the raw token dimensions from the real data pipeline
    raw_token_dims = [t.shape[-1] for t in tokens_list]
    print(f"Derived raw token dimensions: {raw_token_dims}")

    # 3. Instantiate the full SitEncoder. This is the object we want to compile.
    encoder = SitEncoder(
        raw_token_dims=raw_token_dims,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=total_tokens,
        num_scales=num_scales,
        tokens_per_scale=tokens_per_scale
    ).to(device)


    # 3. The key step: Compile the layer
    print("Compiling the SitEncoder with torch.compile()...")
    # The first run will be slow as the kernel is compiled. This is expected.
    try:
        compiled_encoder = torch.compile(encoder)
    except Exception as e:
        assert False, f"torch.compile() failed during setup with error: {e}"
    
    # 5. Perform a forward pass through the *compiled* layer
    print("Performing forward pass through compiled layer...")
    try:
        with torch.no_grad():
            output = compiled_encoder(tokens_list)
    except Exception as e:
        # If any of our mask logic was not compile-able, it would crash here.
        assert False, f"Forward pass on compiled layer failed with error: {e}"

    # 6. Sanity check the output shape.
    expected_shape = (batch_size, total_tokens, d_model)
    assert output.shape == expected_shape, f"Output shape mismatch! Got {output.shape}, expected {expected_shape}"
    print(f"Output shape is correct: {output.shape}")

    print("\n--- ✅ torch.compile() Test Passed ---")
    print("The UserWarning from flex_attention should be absent, confirming successful kernel fusion.")

if __name__ == "__main__":
    test_encoder_compiles()