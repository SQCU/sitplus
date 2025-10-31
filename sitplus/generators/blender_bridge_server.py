# SITPLUS_PYTHON_SCOPE
# sitplus.generators.blender_bridge_server.py
# This is a 'real' Python script managed by uv.
# It runs a WebSocket server and manages the Blender worker subprocess.
import asyncio
import websockets
import subprocess
import json
import os
import io
from setup_blender import get_blender_executable_path
from sitplus.utils.ipc_protocol import (
    try_unpack, PacketType, MAGIC_NUMBER, 
    unpack_teapot_data, pack_teapot_data, pack_jsonrpc
)
import dataclasses
import itertools
from typing import Set

@dataclasses.dataclass
class BlenderWorker:
    process: asyncio.subprocess.Process
    stdin: asyncio.StreamWriter
    stdout: asyncio.StreamReader
    stderr: asyncio.StreamReader
    id: int
    state: str = 'idle' # (e.g., 'idle', 'busy')
    

class BlenderBridge:
    # --- 2. Revised __init__ ---
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.workers: list[BlenderWorker] = []
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.request_counter = itertools.count() # For generating unique request IDs
        self.pending_requests = {} # Maps request_id -> original websocket

                # NEW: State management
        self.worker_map = {} # Maps worker.id -> {"client_ws": ws, "task_id": str}
        self.job_queue = asyncio.Queue() # A single queue for all incoming jobs from all clients


    # --- 3. Revised start() ---
    async def start(self):
        blender_exe = get_blender_executable_path()
        
        for i in range(self.num_workers):
            print(f"Bridge: Starting Blender worker {i}...")
            cmd = [
                blender_exe, 
                "--python", "sitplus/generators/generate_utahs.py",
                "--window-geometry", "0", "0", "1", "1" 
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            worker = BlenderWorker(
                process=process,
                stdin=process.stdin,
                stdout=process.stdout,
                stderr=process.stderr,
                id=i
            )
            self.workers.append(worker)
            
            # Launch dedicated reader tasks for each worker
            asyncio.create_task(self._stdout_reader(worker))
            asyncio.create_task(self._stderr_reader(worker))
            print(f"Bridge: Worker {i} (PID: {process.pid}) is running.")

        # NEW: Start the central dispatcher task
        asyncio.create_task(self._dispatcher())

    # --- 4. Revised stop() ---
    async def stop(self):
        print("Bridge: Stopping all Blender workers...")
        for worker in self.workers:
            if worker.process.returncode is None: # Check if it's still running
                # Send a quit command to allow graceful shutdown
                try:
                    quit_cmd = {"command": "quit"}
                    worker.stdin.write((json.dumps(quit_cmd) + '\n').encode('utf-8'))
                    await worker.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    # The process might have already died
                    pass
                
                # Give it a moment, then terminate if it's still alive
                try:
                    await asyncio.wait_for(worker.process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    print(f"Bridge: Worker {worker.id} did not quit gracefully, terminating.")
                    worker.process.terminate()
                    await worker.process.wait()
        
        print("Bridge: All workers stopped.")

    async def handle_client(self, websocket):
        """Now only queues jobs. The dispatcher handles the rest."""
        print(f"Bridge: Client connected.")
        self.clients.add(websocket)
        try:
            async for message in websocket:
                try:
                    command_data = json.loads(message)
                    # A job is a tuple of (websocket, command_data)
                    await self.job_queue.put((websocket, command_data))
                except json.JSONDecodeError:
                    print(f"Bridge: Received invalid JSON from client: {message}")
        except websockets.exceptions.ConnectionClosed:
            print("Bridge: Client disconnected.")
        finally:
            self.clients.discard(websocket)

    async def _dispatcher(self):
        """The new heart of the bridge. Runs forever, dispatching jobs to idle workers."""
        print("Bridge: Dispatcher started.")
        while True:
            # 1. Get the next job from the queue
            client_ws, command_data = await self.job_queue.get()

            # 2. Find an idle worker (wait if none are available)
            idle_worker = await self._get_idle_worker()
            
            # 3. Assign the job
            task_id = command_data.get("task_id", "unknown")
            print(f"Bridge: Dispatching task '{task_id}' to worker {idle_worker.id}")
            
            idle_worker.state = 'busy'
            self.worker_map[idle_worker.id] = {"client_ws": client_ws, "task_id": task_id}
            
            # 4. Send the command to the worker
            try:
                # The worker only needs the core command, not the task_id
                worker_cmd = command_data['command']
                idle_worker.stdin.write((json.dumps(worker_cmd) + '\n').encode('utf-8'))
                await idle_worker.stdin.drain()
            except Exception as e:
                print(f"Bridge: Error sending job to worker {idle_worker.id}: {e}")
                # Reset worker state if send fails
                idle_worker.state = 'idle'
                self.worker_map.pop(idle_worker.id, None)

    async def _get_idle_worker(self) -> BlenderWorker:
        """Continuously checks for an idle worker, yielding control if all are busy."""
        while True:
            for worker in self.workers:
                if worker.state == 'idle':
                    return worker
            await asyncio.sleep(0.01) # Wait a bit before checking again

    # --- 6. Revised _stdout_reader(worker) (The Collector) ---
    async def _stdout_reader(self, worker: BlenderWorker):
        buffer = b''
        while worker.process.returncode is None:
            try:
                chunk = await worker.stdout.read(4096)
                if not chunk:
                    break # stdout closed
                
                buffer += chunk
                
                while len(buffer) > 0:
                    if not buffer.startswith(MAGIC_NUMBER):
                        # Handle plain text log lines
                        newline_pos = buffer.find(b'\n')
                        if newline_pos == -1: break
                        line = buffer[:newline_pos+1]
                        print(f"BLENDER_{worker.id}_STDOUT: {line.decode('utf-8', errors='ignore').rstrip()}")
                        buffer = buffer[newline_pos+1:]
                        continue

                    # We have a potential packet
                    result = try_unpack(buffer)
                    if result is None:
                        break # Incomplete packet, wait for more data
                        
                    packet_type, payload, bytes_consumed = result
                    
                    # Process the complete packet
                    if packet_type == PacketType.DATA:
                        params, image_bytes = unpack_teapot_data(payload)
                        request_id = params.get('request_id')
                        
                        # Route the response back to the correct client
                        if request_id in self.pending_requests:
                            client_ws = self.pending_requests.pop(request_id)
                            response_message = pack_teapot_data(params, image_bytes)
                            if client_ws.open:
                                await client_ws.send(response_message)
                        else:
                            print(f"Bridge: Received data for unknown request_id: {request_id}")
                    if packet_type == PacketType.DATA:
                        # Use the WORKER'S ID to get the correct task info.
                        # Do NOT rely on the worker to send back the request_id.
                        if worker.id in self.worker_map:
                            request_info = self.worker_map.pop(worker.id)
                            client_ws = request_info["client_ws"]
                            task_id = request_info["task_id"]
                            # 2. Unpack the data payload from the worker.
                            params, image_bytes = unpack_teapot_data(payload)
                            # 3. Add the task_id to the params FOR THE CLIENT.
                            params['task_id'] = task_id
                            # 4. Create the final response packet for the client.
                            response_message = pack_teapot_data(params, image_bytes)
                            # 5. Check connection with the CORRECT attribute and send.
                            try:
                                await client_ws.send(response_message)
                            except websockets.exceptions.ConnectionClosed:
                                print(f"Bridge: Client for task '{task_id}' disconnected before data could be sent.")
                            # 6. IMPORTANT: Mark the worker as idle again.
                            worker.state = 'idle'
                        else:
                            print(f"Bridge ERROR: Received data from unmapped/idle worker {worker.id}. Discarding.")
                        # --- END FIX ---
                    elif packet_type == PacketType.LOG:
                        print(f"BLENDER_{worker.id}_LOG: {payload.decode('utf-8')}")
                    elif packet_type == PacketType.ERROR:
                        print(f"BLENDER_{worker.id}_ERROR: {payload.decode('utf-8')}")
                    
                    buffer = buffer[bytes_consumed:]
            
            except Exception as e:
                print(f"Bridge: Error reading stdout from worker {worker.id}: {e}")
                worker.state = 'idle' 
                self.worker_map.pop(worker.id, None)
                break
        print(f"Bridge: Stdout reader for worker {worker.id} finished.")

    # --- 7. Revised _stderr_reader(worker) (for per-worker logging) ---
    async def _stderr_reader(self, worker: BlenderWorker):
        while worker.process.returncode is None:
            try:
                line = await worker.stderr.readline()
                if not line:
                    break
                print(f"BLENDER_{worker.id}_STDERR: {line.decode('utf-8', errors='ignore').rstrip()}")
            except Exception as e:
                print(f"Bridge: Error reading stderr from worker {worker.id}: {e}")
                break
        print(f"Bridge: Stderr reader for worker {worker.id} finished.")
        """Broadcast error message to clients (optional feature)."""
        print(f"ERROR: {message}")


# --- 8. Revised main() function ---
async def main():
    # Set number of workers (e.g., half the number of CPU cores)
    num_workers = max(1, os.cpu_count() // 2)
    print(f"Bridge: Initializing with {num_workers} workers.")
    
    bridge = BlenderBridge(num_workers=num_workers)
    await bridge.start()
    
    host = "127.0.0.1"
    port = 8765
    print(f"Bridge: Starting WebSocket server on ws://{host}:{port}")
    
    try:
        async with websockets.serve(
            bridge.handle_client, 
            host, 
            port,
            max_size=100 * 1024 * 1024
        ):
            await asyncio.Future()  # Run forever
    finally:
        await bridge.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBridge: Shutting down.")