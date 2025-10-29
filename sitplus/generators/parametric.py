# sitplus/generators/parametric.py
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# a debug of variations
import random
from torchvision.transforms.functional import affine

def render_text_image(
    text: str = "Wavelet",
    size: tuple[int, int] = (256, 256),
    font_size: int = 50,
    background_color: str = "black",
    text_color: str = "white",
) -> torch.Tensor:
    """
    Generates a PyTorch tensor of a rendered text image.

    Args:
        text: The text to render.
        size: The (width, height) of the image.
        font_size: The font size.
        background_color: The background color.
        text_color: The text color.

    Returns:
        A torch.Tensor of shape (1, 3, H, W) with values in [0, 1].
    """
    # Create a blank image
    image = Image.new("RGB", size, background_color)
    draw = ImageDraw.Draw(image)

    # Load a font (try to find a common one, fallback to default)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text position for centering
    # Use textbbox for more accurate centering
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((size[0] - text_width) / 2, (size[1] - text_height) / 2)

    # Draw the text
    draw.text(position, text, fill=text_color, font=font)

    # Convert to numpy array and then to torch tensor
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)

    return image_tensor

# --- ADD THIS NEW FUNCTION ---
def generate_image_variations(
    base_image: torch.Tensor,
    num_variations: int = 1000
):
    """
    Yields a series of transformed variations of a base image.

    Args:
        base_image: The starting tensor of shape (1, C, H, W).
        num_variations: The number of variations to generate.
    """
    for _ in range(num_variations):
        # Generate random affine transformation parameters
        angle = random.uniform(-30, 30)
        translate_x = random.randint(-20, 20)
        translate_y = random.randint(-20, 20)
        scale = random.uniform(0.9, 1.1)
        shear = random.uniform(-10, 10)

        # Apply the affine transformation
        transformed_image = affine(
            base_image,
            angle=angle,
            translate=(translate_x, translate_y),
            scale=scale,
            shear=[shear]
        )
        yield transformed_image

if __name__ == '__main__':
    # A simple self-test to save the generated image
    print("Generating a test image: 'parametric_test.png'")
    test_tensor = render_text_image("Test Recon", size=(512, 256))

    # Convert back to PIL for saving
    from torchvision.utils import save_image
    save_image(test_tensor, "parametric_test.png")
    print("Saved.")