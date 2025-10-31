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

class BlenderBridge:
    def __init__(self):
        self.process = None
        self.stdout_reader = None
        self.stderr_reader = None
        self.writer = None
        self.clients = set()

    async def start(self):
        blender_exe = get_blender_executable_path()
        cmd = [
            blender_exe, 
            "--python", "sitplus/generators/generate_utahs.py",
            # Add minimal window flags to satisfy OpenGL/GUI Context requirements
            "--window-geometry", "0", "0", "1", "1" 
        ]
        
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        self.stdout_reader = self.process.stdout
        self.stderr_reader = self.process.stderr
        self.writer = self.process.stdin
        
        # Start background tasks to read stdout/stderr
        asyncio.create_task(self._stdout_reader())
        asyncio.create_task(self._stderr_reader())
        print("Bridge: Blender worker process started.")

    async def stop(self):
        if self.process:
            self.process.terminate()
            await self.process.wait()
            print("Bridge: Blender worker process stopped.")

    async def handle_client(self, websocket):
        print(f"Bridge: Client connected.")
        self.clients.add(websocket)
        try:
            async for message in websocket:
                # Forward command to Blender worker
                self.writer.write((message + '\n').encode('utf-8'))
                await self.writer.drain()
                
        except websockets.exceptions.ConnectionClosed:
            print("Bridge: Client disconnected.")
        finally:
            self.clients.discard(websocket)

    async def _stdout_reader(self):
        """Reads the packet stream from Blender's stdout."""
        buffer = b''
        while True:
            try:
                # Read available data
                chunk = await self.stdout_reader.read(4096)
                if not chunk:
                    print("Bridge: Blender stdout closed.")
                    break
                
                buffer += chunk
                
                # Try to parse packets from the buffer
                while len(buffer) > 0:
                    # Check if this looks like a packet
                    if len(buffer) >= 4 and buffer[:4] == MAGIC_NUMBER:
                        # Try to parse as packet
                        result = try_unpack(buffer)
                        
                        if result is None:
                            # Valid packet header but incomplete, wait for more data
                            break
                        
                        packet_type, payload, bytes_consumed = result
                        await self.process_packet(packet_type, payload)
                        buffer = buffer[bytes_consumed:]
                    else:
                        # Not a packet, find the next newline and output as plain text
                        newline_pos = buffer.find(b'\n')
                        if newline_pos == -1:
                            # No newline yet, wait for more data
                            break
                        
                        # Output the line as plain text
                        line = buffer[:newline_pos+1]
                        print(f"BLENDER_STDOUT: {line.decode('utf-8', errors='ignore').rstrip()}")
                        buffer = buffer[newline_pos+1:]
                        
            except Exception as e:
                print(f"Bridge: Error reading stdout: {e}")
                import traceback
                traceback.print_exc()
                break
    
    def _try_parse_packet(self, buffer):
        """
        Attempts to parse one packet from the buffer.
        Returns (packet_type, payload, bytes_consumed) or None if incomplete.
        Assumes buffer starts with MAGIC_NUMBER (already checked by caller).
        """
        import struct
        import zlib
        
        if len(buffer) < HEADER_SIZE:
            return None  # Incomplete header
        
        # Parse header
        try:
            magic, version, packet_type, payload_len, checksum = struct.unpack(
                HEADER_FORMAT, buffer[:HEADER_SIZE]
            )
        except struct.error:
            print(f"Bridge: Failed to unpack header")
            return None
        
        # Sanity checks (magic already verified by caller)
        if version != PROTOCOL_VERSION:
            print(f"Bridge: Bad protocol version: {version}")
            return None
        
        # Check if we have the full payload
        total_size = HEADER_SIZE + payload_len
        if len(buffer) < total_size:
            return None  # Need more data
        
        # Extract payload
        payload = buffer[HEADER_SIZE:total_size]
        
        # Verify checksum
        header_for_check = struct.pack('!4sBBI', magic, version, packet_type, payload_len)
        calculated_checksum = zlib.crc32(header_for_check + payload)
        
        if checksum != calculated_checksum:
            print(f"Bridge: Checksum mismatch!")
            return None
        
        return (packet_type, payload, total_size)

    async def _stderr_reader(self):
        """Reads stderr for debugging output."""
        while True:
            try:
                line = await self.stderr_reader.readline()
                if not line:
                    break
                print(f"BLENDER_STDERR: {line.decode('utf-8', errors='ignore').rstrip()}")
            except Exception as e:
                print(f"Bridge: Error reading stderr: {e}")
                break
    
    async def process_packet(self, packet_type, payload):
        """Translates a parsed packet into a WebSocket message."""
        if packet_type == PacketType.DATA:
            try:
                # Use the clean protocol function
                params, image_bytes = unpack_teapot_data(payload)
                await self.broadcast_data(params, image_bytes)
            except Exception as e:
                print(f"Bridge: Error processing DATA packet: {e}")
                import traceback
                traceback.print_exc()
                
        elif packet_type == PacketType.LOG:
            message = payload.decode('utf-8')
            print(f"BLENDER_LOG: {message}")
            
        elif packet_type == PacketType.ERROR:
            message = payload.decode('utf-8')
            print(f"BLENDER_ERROR: {message}")
        
    async def broadcast_data(self, params, image_bytes):
        """Send data to all connected clients."""
        if not self.clients:
            return
        
        # PATCH: Use the now-refactored pack_teapot_data function 
        # to create the clean JSON/Base64 payload for the WebSocket.
        message = pack_teapot_data(params, image_bytes)
        
        # Broadcast to all connected clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message) # 'message' is now clean JSON bytes
            except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            # Clean up disconnected clients
            self.clients -= disconnected
    
    async def broadcast_log(self, message):
        """Broadcast log message to clients (optional feature)."""
        print(f"LOG: {message}")
    
    async def broadcast_error(self, message):
        """Broadcast error message to clients (optional feature)."""
        print(f"ERROR: {message}")


async def main():
    bridge = BlenderBridge()
    await bridge.start()
    
    host = "127.0.0.1"
    port = 8765
    print(f"Bridge: Starting WebSocket server on ws://{host}:{port}")
    
    try:
        # Increase max message size to 100MB to handle large images
        async with websockets.serve(
            bridge.handle_client, 
            host, 
            port,
            max_size=100 * 1024 * 1024  # 100MB limit
        ):
            await asyncio.Future()  # Run forever
    finally:
        await bridge.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBridge: Shutting down.")