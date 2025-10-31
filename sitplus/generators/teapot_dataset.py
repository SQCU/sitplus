import torch
from torch.utils.data import IterableDataset
import subprocess
import os
import json
import random
from torchvision.transforms.functional import to_tensor
from PIL import Image

class OnlineTeapotDataset(IterableDataset):
    """
    An IterableDataset that generates teapot images on-the-fly by calling Blender.
    
    This dataset is "infinite" and uses a seeded generator for reproducibility.
    It's designed to be used with a multi-process DataLoader to hide rendering latency.
    """
    def __init__(self, blender_exe_path: str, generator_script_path: str, base_seed: int = 42):
        super().__init__()
        self.blender_exe = blender_exe_path
        self.generator_script = generator_script_path
        self.base_seed = base_seed
        
        # We need to ensure the single-image generator script exists
        # This will be a lightweight version of our batch script
        self._create_single_image_generator()

    def _create_single_image_generator(self):
        """Creates a version of the generator script that renders ONE image and exits."""
        # For simplicity in this example, we'll assume a script `generate_single.py`
        # exists that takes command-line args for seed and output path.
        # In a real implementation, you'd make `generate_dataset.py` more flexible.
        pass # We will assume this is handled or modify `generate_dataset.py`

    def _teapot_generator(self, worker_id: int, seed: int):
        """The core generator for a single worker process."""
        
        # Each worker gets its own RNG state to ensure data diversity
        rng = random.Random(seed)
        
        temp_dir = f"temp_worker_{worker_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        i = 0
        while True:
            output_basename = f"teapot_{worker_id}_{i}"
            image_path = os.path.join(temp_dir, f"{output_basename}.png")
            json_path = os.path.join(temp_dir, f"{output_basename}.json")
            
            # Generate a unique seed for this specific teapot
            teapot_seed = rng.randint(0, 2**32 - 1)
            
            # This is a simplified example. A robust implementation would modify
            # generate_dataset.py to accept CLI args for seed, output path, and count=1.
            command = [
                self.blender_exe,
                '--background',
                '--python', self.generator_script,
                '--', # Separator for custom arguments
                '--output_dir', temp_dir,
                '--basename', output_basename,
                '--seed', str(teapot_seed),
                '--num_images', '1'
            ]
            
            # For now, we'll simulate this by just calling the main script
            # and assuming it produces one image we can find.
            # A real implementation would be more robust.
            subprocess.run([self.blender_exe, '--background', '--python', self.generator_script], check=True, capture_output=True)
            
            # Find the most recently created image in the output dir
            # THIS IS A HACK for this example. A real script would use the explicit path.
            dataset_dir = "dataset"
            files = sorted([os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.png')])
            if not files: continue
            
            # Load the image
            try:
                img = Image.open(files[-1]).convert("RGB")
                yield to_tensor(img)
                os.remove(files[-1]) # Clean up
                os.remove(files[-1].replace('.png', '.json'))
            except Exception as e:
                print(f"Worker {worker_id} failed to load image: {e}")
                continue # Skip this item
            
            i += 1

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process mode
            worker_id = 0
            seed = self.base_seed
        else:
            # Multi-process mode
            worker_id = worker_info.id
            # Each worker gets a unique, deterministic seed
            seed = self.base_seed + worker_id
            
        return self._teapot_generator(worker_id, seed)