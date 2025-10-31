# tests.benchmark_teapot_hertz
# python -m tests.benchmark_teapot_hertz --detached-blendserv
# for running in double terminal blender backen debugging mode; elsewise:
# python -m tests.benchmark_teapot_hertz
import torch
import numpy as np
import time
import subprocess
import json
import base64
import os
import asyncio
import websockets
import argparse
from setup_blender import get_blender_executable_path
from sitplus.utils.ipc_protocol import unpack_teapot_data

def validate_render_not_blank(image_tensor: torch.Tensor, min_mean: float = 0.05, min_variance: float = 0.001) -> bool:
    """
    Stateless validator: checks if a render is actually rendered or just blank/grey.
    
    Args:
        image_tensor: (C, H, W) tensor, expected to be RGB in [0, 1] range
        min_mean: Minimum mean brightness (default 0.05, very permissive)
        min_variance: Minimum variance (default 0.001, very permissive)
    
    Returns:
        True if image appears valid, False if likely blank/grey/zeros
    """
    # Check 1: All zeros (complete failure)
    if torch.all(image_tensor == 0):
        print(f"  Validation fail: all zeros")
        return False
    
    # Check 2: All ones (white screen)
    if torch.all(image_tensor == 1):
        print(f"  Validation fail: all ones")
        return False
    
    # Check 3: Basic statistics
    mean_val = torch.mean(image_tensor).item()
    variance = torch.var(image_tensor).item()
    min_val = torch.min(image_tensor).item()
    max_val = torch.max(image_tensor).item()
    
    print(f"  Stats: mean={mean_val:.4f}, var={variance:.4f}, min={min_val:.4f}, max={max_val:.4f}")
    
    # Check 4: Variance threshold (very permissive)
    if variance < min_variance:
        print(f"  Validation fail: variance {variance:.4f} < {min_variance}")
        return False
    
    # Check 5: Not completely dark
    if mean_val < min_mean:
        print(f"  Validation fail: mean {mean_val:.4f} < {min_mean}")
        return False
    
    # Check 6: Has some variation in values
    value_range = max_val - min_val
    if value_range < 0.01:
        print(f"  Validation fail: range {value_range:.4f} too small")
        return False
    
    print(f"  Validation PASS")
    return True

async def benchmark_hertz(websocket, resolution, num_images):
    """
    Runs a single benchmark on a connected WebSocket.
    
    The server process and connection management are handled by the caller (main).
    """
    print(f"\n--- Benchmarking {num_images} images at {resolution[0]}x{resolution[1]} ---")
    
    image_cache = []
    start_time = time.perf_counter()
    
    # We use the existing, persistent websocket connection
    
    for i in range(num_images):
        # A. Send command
        command = { "command": "render", "resolution_x": resolution[0], "resolution_y": resolution[1] }
        await websocket.send(json.dumps(command))
        
        # B. Receive response (This is where the previous trace failed on the second run)
        try:
            response = await websocket.recv()
        except websockets.exceptions.ConnectionClosedOK:
            print("\nERROR: Server closed connection prematurely.")
            break

        # Parse the JSON response (client logic assumes clean JSON/Base64 from the fixed bridge)
        if isinstance(response, bytes):
            try:
                data = json.loads(response.decode('utf-8'))
                params = data["params"]
                image_bytes = base64.b64decode(data["image"])
            except json.JSONDecodeError:
                print(f"ERROR: Failed to decode JSON response: {response[:100]}...")
                break
        else:
            print(f"ERROR: Unexpected response format: {type(response)}")
            break

        if not image_bytes:
            print(f"ERROR: Server signaled a failure")
            break

        # PATCH: Use the resolution returned by the server for reshaping.
        # This fixes the bug where the client assumes success before checking the data.
        if 'render' not in params:
             print("ERROR: Server response missing 'render' metadata.")
             break
        
        recv_res_y = params['render']['resolution_y']
        recv_res_x = params['render']['resolution_x']

        # Validation: Check if the buffer size matches the metadata
        expected_size = recv_res_y * recv_res_x * 4 * 4 # Height * Width * Channels * sizeof(float32)
        if len(image_bytes) != expected_size:
            print(f"ERROR: Buffer size mismatch! Metadata: {recv_res_x}x{recv_res_y} ({expected_size} bytes) vs Actual: {len(image_bytes)} bytes")
            # If the size is the old 256x256 size, it confirms the backend bug.
            break

        # C. Convert to tensor
        image_np = np.frombuffer(image_bytes, dtype=np.float32).copy().reshape(recv_res_y, recv_res_x, 4)
        image_tensor = torch.from_numpy(image_np[:, :, :3]).permute(2, 0, 1)
        if not validate_render_not_blank(image_tensor):
            print(f"WARNING: Render {i} appears blank/invalid, not that you can do anything about it.")

        image_cache.append(image_tensor)
        
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    hertz = len(image_cache) / duration if duration > 0 else 0
    print(f"Generated {len(image_cache)} images in {duration:.2f} seconds.")
    print(f"Effective Teapot-Hertz: {hertz:.2f} images/sec")
    return hertz


async def main():
    parser = argparse.ArgumentParser(description="Teapot-Hertz Benchmark Client.")
    parser.add_argument(
        '--detached-blendserv', 
        action='store_true', 
        help='Do not launch the Blender server process. Assume it is running externally.'
    )
    args = parser.parse_args()
    
    resolutions_to_test = [
        (256, 256),
        (320, 768), 
        (512, 512),
    ]
    num_images_per_test = 100
    
    server_process = None
    uri = "ws://127.0.0.1:8765"
    
    # --- 1. Server Launch/Management ---
    if not args.detached_blendserv:
        blender_exe = get_blender_executable_path()
        # CORRECTED LAUNCH SCRIPT: Use blender_bridge_server.py
        server_cmd = [blender_exe, "--background", "--python", "sitplus/generators/blender_bridge_server.py"]
        
        print("Launching Blender Bridge Server...")
        server_process = await asyncio.create_subprocess_exec(
            *server_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        print(f"Server PID: {server_process.pid}. Waiting for connection...")
    else:
        print("Running in detached mode. Assuming Blender Bridge Server is running externally.")
    # --- END Server Launch/Management ---

    # --- 2. Client Connection ---
    # The client must connect regardless of detached mode
    for i in range(10): # Try for 10 seconds
        try:
            # Connect the client once
            websocket = await websockets.connect(
                uri, 
                max_size=100 * 1024 * 1024, 
                open_timeout=2
            )
            print("Server connected.")
            break
        except (ConnectionRefusedError, asyncio.TimeoutError):
            if i == 9 and not args.detached_blendserv:
                # If we launched it but failed to connect, terminate the process.
                server_process.terminate()
                await server_process.wait()
            await asyncio.sleep(1)
    else:
        raise RuntimeError(f"Failed to connect to Blender WebSocket server at {uri}.")
    # --- END Client Connection ---

    results = {}
    
    try:
        # --- 3. Sequential Benchmarks ---
        for res in resolutions_to_test:
            # Pass the single websocket connection
            hertz = await benchmark_hertz(websocket, res, num_images_per_test)
            results[f"{res[0]}x{res[1]}"] = hertz
            
    finally:
        # --- 4. Single Shutdown (AFTER all tests) ---
        if 'websocket' in locals():
            # Attempt to send the quit command and rely on exception handling if it's closed.
            try:
                print("\nSending 'quit' command to server...")
                await websocket.send(json.dumps({"command": "quit"}))
                
                # Wait for the server to close the connection
                await websocket.recv()
            except websockets.exceptions.ConnectionClosed:
                pass # Connection successfully closed (by server or client)
            except Exception as e:
                # Log any other unexpected exception during final send (e.g., Timeout)
                print(f"Warning during final shutdown send: {e}")
            
            # Close the client side connection explicitly
            try:
                await websocket.close()
            except Exception:
                pass

        # Terminate the process if we launched it
        if server_process and server_process.returncode is None:
            print("Terminating Blender Bridge Server process...")
            server_process.terminate()
            await server_process.wait()
            
    print("\n--- Final Teapot-Hertz Summary ---")
    for res_str, hertz in results.items():
        print(f"  {res_str}: {hertz:.2f} Hz")
    print("----------------------------------")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark: Shutting down.")