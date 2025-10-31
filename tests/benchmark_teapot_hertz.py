# tests.benchmark_teapot_hertz
import torch
import numpy as np
import time
import subprocess
import json
import base64
import os
import asyncio
import websockets
from setup_blender import get_blender_executable_path
from sitplus.utils.ipc_protocol import unpack_teapot_data


async def benchmark_hertz(resolution, num_images):
    print(f"\n--- Benchmarking {num_images} images at {resolution[0]}x{resolution[1]} ---")
    
    blender_exe = get_blender_executable_path()
    server_cmd = [blender_exe, "--background", "--python", "sitplus/generators/generate_utahs.py"]
    
    # Launch the server. We don't need its stdin/stdout anymore. We'll just watch its logs.
    server_process = await asyncio.create_subprocess_exec(
        *server_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Wait for the server to be ready by trying to connect
    uri = "ws://127.0.0.1:8765"
    for _ in range(10): # Try for 10 seconds
        try:
            async with websockets.connect(uri, max_size=100 * 1024 * 1024  # 100MB limit
        ) as websocket:
                print("Server connected.")
                break
        except ConnectionRefusedError as e:
            print(f"connref:{e}")
            await asyncio.sleep(1)
    else:
        raise RuntimeError("Failed to connect to Blender WebSocket server.")

    async with websockets.connect(uri, max_size=100 * 1024 * 1024  # 100MB limit
        ) as websocket:
        image_cache = []
        start_time = time.perf_counter()
        
        for i in range(num_images):
            # A. Send command
            command = { "command": "render", "resolution_x": resolution[0], "resolution_y": resolution[1] }
            await websocket.send(json.dumps(command))
            
            # B. Receive response
            response = await websocket.recv()

            # Parse the JSON response
            if isinstance(response, bytes):
                # This is assuming the bridge is fixed to send the JSON/Base64 payload directly
                params, image_bytes = unpack_teapot_data(response)
            else:
                print(f"ERROR: Unexpected response format: {type(response)}")
                break

            if not image_bytes:
                print(f"ERROR: Server signaled a failure")
                break

            # C. Convert to tensor
            image_np = np.frombuffer(image_bytes, dtype=np.float32).reshape(resolution[1], resolution[0], 4)
            image_tensor = torch.from_numpy(image_np[:, :, :3]).permute(2, 0, 1)
            image_cache.append(image_tensor)
            
        end_time = time.perf_counter()

        # Cleanly shut down the server
        await websocket.send(json.dumps({"command": "quit"}))

    # Terminate the process
    server_process.terminate()
    await server_process.wait()
    
    duration = end_time - start_time
    hertz = num_images / duration
    print(f"Generated {len(image_cache)} images in {duration:.2f} seconds.")
    print(f"Effective Teapot-Hertz: {hertz:.2f} images/sec")
    return hertz


async def main():
    resolutions_to_test = [
        (256, 256),
        (320, 768), # Flipped for a portrait aspect ratio
        (512, 512),
        # (1024, 2048), # This might be very slow or run out of VRAM
    ]
    num_images_per_test = 100
    
    results = {}
    for res in resolutions_to_test:
        hertz = await benchmark_hertz(res, num_images_per_test)
        results[f"{res[0]}x{res[1]}"] = hertz
        
    print("\n--- Final Teapot-Hertz Summary ---")
    for res_str, hertz in results.items():
        print(f"  {res_str}: {hertz:.2f} Hz")
    print("----------------------------------")

if __name__ == "__main__":
    asyncio.run(main())