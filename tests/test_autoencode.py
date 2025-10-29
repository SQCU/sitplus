# python -m tests.test_autoencode
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor, resize
from torchvision.utils import save_image
from PIL import Image
import requests
import io

from sitplus.model import SitAutoencoder

def test_untrained_autoencoder_intervention():
    print("--- Running Untrained Autoencoder Intervention Test ---")
    
    # 1. Config and Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'image_size': 256,
        'num_dwt_levels': 4,
        'tokens_per_scale_edge': 8,
        'd_model': 256,
        'n_layers': 6,
        'n_heads': 8,
        # Derived values that are useful to have
        'tokens_per_scale': 64, # 8*8
        'num_scales': 5,        # 4 levels + 1 approx
    }
    
    config['tokens_per_scale'] = config['tokens_per_scale_edge']**2
    config['num_scales'] = config['num_dwt_levels'] + 1
    
    # 2. Load a real image (the Utah Teapot)
    try:
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Original_Utah_Teapot.jpg/1024px-Original_Utah_Teapot.jpg"
        response = requests.get(url)
        img_pil = Image.open(io.BytesIO(response.content)).convert("RGB")
        img_tensor = to_tensor(img_pil).unsqueeze(0)
        # Resize and crop to a square
        img_tensor = resize(img_tensor, [config['image_size'], config['image_size']], antialias=True)
    except Exception as e:
        print(f"Failed to download test image, using random tensor. Error: {e}")
        img_tensor = torch.randn(1, 3, config['image_size'], config['image_size'])
    
    original_image = img_tensor.to(device)
    
    # 3. Instantiate and initialize the model
    model = SitAutoencoder(config).to(device)
    print("Model instantiated and initialized with Kaiming weights.")
    
    # 4. Control Pass: Autoencode the image
    with torch.no_grad():
        control_recon, control_latents = model(original_image)
    
    # 5. Probe and Visualize Control Latents
    print("Probing control latents...")
    # Let's visualize scale 2 (the middle detail scale)
    probe_scale_idx = 2
    probe_start = probe_scale_idx * config['tokens_per_scale']
    probe_end = probe_start + config['tokens_per_scale']
    
    scale_2_latents = control_latents[:, probe_start:probe_end, :]
    # Get L2 norm across the feature dim and reshape to 8x8
    latent_vis = scale_2_latents.norm(dim=-1).reshape(config['tokens_per_scale_edge'], config['tokens_per_scale_edge'])
    plt.imsave("probe_control_latents.png", latent_vis.cpu().numpy(), cmap='viridis')
    print("Saved 'probe_control_latents.png'.")
    
    # 6. Intervention: Draw a smiley face in the latents of the same scale
    print("Intervening on latents to draw a smiley face...")
    intervened_latents = control_latents.clone()
    
    # Reshape the target scale for easy 2D indexing
    scale_2_latents_reshaped = intervened_latents[:, probe_start:probe_end, :].reshape(config['tokens_per_scale_edge'], config['tokens_per_scale_edge'], -1)
    
    # Smiley face coords on an 8x8 grid
    smiley_coords = [(2, 2), (2, 5), (5, 2), (5, 3), (5, 4), (5, 5)] # Eyes and a smile
    for y, x in smiley_coords:
        scale_2_latents_reshaped[y, x, :] = 5.0 # Set to a high, constant value
        
    # 7. Intervention Pass: Decode the modified latents
    with torch.no_grad():
        intervention_recon = model.decode(intervened_latents)

    # --- THIS IS THE FIX / DEBUG STEP ---
    # Print the shapes to see what's going wrong before we save.
    print(f"Shape of original_image: {original_image.shape}")
    print(f"Shape of control_recon: {control_recon.shape}")
    print(f"Shape of intervention_recon: {intervention_recon.shape}")
    # --- END FIX ---

    # 8. Save all images for comparison
    output_batch = torch.cat([
        original_image.clamp(0, 1), 
        control_recon.clamp(0, 1), 
        intervention_recon.clamp(0, 1)
    ], dim=0) # Concatenate along the batch dimension
    
    save_image(
        output_batch,
        "test_autoencoder_results.png",
        nrow=3, # Display all 3 images in a single row
        normalize=False # We already clamped, no need to normalize again
    )
    print("Saved 'test_autoencoder_results.png' with [Original | Control Recon | Intervention Recon].")
    print("\n--- ✅ Autoencoder Test Finished ---")
    print("Check the output images. 'Control' should look like random noise. 'Intervention' should be noise with a ghostly pattern.")

if __name__ == "__main__":
    test_untrained_autoencoder_intervention()