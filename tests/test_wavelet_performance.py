# python -m tests.test_wavelet_performance

import torch
import time
from pytorch_wavelets import DWTForward, DWTInverse

from sitplus.generators.parametric import render_text_image, generate_image_variations

def benchmark_batched_device(device_str: str, base_image: torch.Tensor, total_images: int, batch_size: int, levels: int, wavelet_type: str):
    """
    Runs a batched DWT/IDWT benchmark on a specified device and returns the elapsed time.
    """
    print(f"\n--- Benchmarking on device: {device_str.upper()} (Batch Size: {batch_size}) ---")
    device = torch.device(device_str)
    num_batches = total_images // batch_size

    # 1. Initialize models and move to the target device
    dwt = DWTForward(J=levels, wave=wavelet_type, mode='zero').to(device)
    idwt = DWTInverse(wave=wavelet_type, mode='zero').to(device)

    # 2. Generate all data on the CPU first to not contaminate the timing loop
    print("Generating all image variations on CPU...")
    # This list will hold batches of tensors
    image_batches = []
    # Create a generator for all images
    variation_generator = generate_image_variations(base_image, total_images)
    for i in range(num_batches):
        # Build one batch
        batch = [next(variation_generator) for _ in range(batch_size)]
        # Stack into a single tensor and add to our list
        image_batches.append(torch.cat(batch, dim=0))
    
    # 3. Warmup phase
    print("Warming up...")
    warmup_batch = image_batches[0].to(device)
    _ = idwt(dwt(warmup_batch))
    # Synchronize after warmup on CUDA to ensure it's complete
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 4. Run and time the main loop
    print(f"Processing {num_batches} batches of size {batch_size}...")
    start_time = time.perf_counter()

    for batch_cpu in image_batches:
        # Single, efficient transfer of the entire batch to the target device
        batch_on_device = batch_cpu.to(device)
        
        # Single, efficient execution on the whole batch
        yl, yh = dwt(batch_on_device)
        _ = idwt((yl, yh))

    # For CUDA, it's crucial to synchronize before stopping the timer
    # to ensure all asynchronous operations have completed.
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    duration = end_time - start_time
    
    images_per_sec = total_images / duration
    print(f"Completed in {duration:.4f} seconds ({images_per_sec:,.2f} images/sec).")
    return duration

if __name__ == "__main__":
    # --- Configuration ---
    TOTAL_IMAGES = 1024  # Use a power of 2 for clean batching
    BATCH_SIZE = 32
    IMAGE_SIZE = (256, 256)
    LEVELS = 4
    WAVELET_TYPE = 'haar'

    print("--- Starting Wavelet Performance Benchmark (Batched) ---")
    
    # Generate a single base image on the CPU
    base_image_cpu = render_text_image(text="Recon", size=IMAGE_SIZE)

    # --- Run Benchmarks ---
    cpu_time = benchmark_batched_device('cpu', base_image_cpu, TOTAL_IMAGES, BATCH_SIZE, LEVELS, WAVELET_TYPE)

    gpu_time = None
    if torch.cuda.is_available():
        gpu_time = benchmark_batched_device('cuda', base_image_cpu, TOTAL_IMAGES, BATCH_SIZE, LEVELS, WAVELET_TYPE)
    else:
        print("\nCUDA not available, skipping GPU benchmark.")

    # --- Report Results ---
    print("\n--- Benchmark Summary ---")
    print(f"Total Images: {TOTAL_IMAGES}, Batch Size: {BATCH_SIZE}")
    print(f"CPU Time: {cpu_time:.4f} seconds")
    if gpu_time is not None:
        print(f"GPU Time: {gpu_time:.4f} seconds")
        speedup = cpu_time / gpu_time
        print(f"\nConclusion: GPU is {speedup:.2f}x faster than CPU with proper batching.")
    print("-------------------------")