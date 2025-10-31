# sitplus.utils.ipc_protocol.py
# sitplus.utils.ipc_protocol.py
"""
Clean IPC protocol for Blender ↔ Bridge ↔ Client communication.

Two packet types:
1. BytesPacket: Raw binary data (logs, errors, or RPC responses)
2. JsonRpcPacket: Structured JSON-RPC for commands

All packets share a common header format for multiplexing over stdout.

PROTOCOL CONSTANTS:
- Magic number: b'TPOT' (identifies packets in mixed stdout)
- Delimiter: NEVER USED - we use proper JSON serialization
- All binary data is base64-encoded within JSON structures
"""

import struct
import zlib
import json
import base64
from typing import Optional, Tuple, Any

# Protocol constants
MAGIC_NUMBER = b'TPOT'
PROTOCOL_VERSION = 0x01

# Packet types
class PacketType:
    """Packet type identifiers."""
    LOG = 0x01      # Human-readable log message (UTF-8 text)
    ERROR = 0x02    # Human-readable error message (UTF-8 text)
    DATA = 0x03     # Binary data blob (JSON with base64-encoded image)
    JSONRPC = 0x04  # JSON-RPC formatted message

# Header format: Magic(4s), Version(B), Type(B), PayloadLen(I), Checksum(I)
HEADER_FORMAT = '!4sBBII'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


# ============================================================================
# LOW-LEVEL PACKET FUNCTIONS (used by Blender worker)
# ============================================================================

def pack(packet_type: int, payload: bytes) -> bytes:
    """
    Creates a complete packet with header, checksum, and payload.
    
    Args:
        packet_type: One of PacketType constants
        payload: Raw bytes to send
    
    Returns:
        Complete packet ready to write to stdout
    """
    # Build header without checksum
    header_no_checksum = struct.pack(
        '!4sBBI', 
        MAGIC_NUMBER, 
        PROTOCOL_VERSION, 
        packet_type, 
        len(payload)
    )
    
    # Calculate checksum over header + payload
    checksum = zlib.crc32(header_no_checksum + payload)
    
    # Build complete header with checksum
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC_NUMBER,
        PROTOCOL_VERSION,
        packet_type,
        len(payload),
        checksum
    )
    
    return header + payload


def try_unpack(buffer: bytes) -> Optional[Tuple[int, bytes, int]]:
    """
    Attempts to parse one packet from a buffer.
    
    Args:
        buffer: Byte buffer that may contain a packet
    
    Returns:
        (packet_type, payload, bytes_consumed) if successful, None otherwise
    """
    if len(buffer) < HEADER_SIZE:
        return None
    
    # Check magic number first (fast rejection of non-packets)
    if buffer[:4] != MAGIC_NUMBER:
        return None
    
    # Parse header
    try:
        magic, version, packet_type, payload_len, checksum = struct.unpack(
            HEADER_FORMAT, 
            buffer[:HEADER_SIZE]
        )
    except struct.error:
        return None
    
    # Validate version
    if version != PROTOCOL_VERSION:
        return None
    
    # Check if we have full payload
    total_size = HEADER_SIZE + payload_len
    if len(buffer) < total_size:
        return None  # Incomplete packet
    
    # Extract payload
    payload = buffer[HEADER_SIZE:total_size]
    
    # Verify checksum
    header_for_check = struct.pack('!4sBBI', magic, version, packet_type, payload_len)
    calculated_checksum = zlib.crc32(header_for_check + payload)
    
    if checksum != calculated_checksum:
        return None  # Corrupted packet
    
    return (packet_type, payload, total_size)


# ============================================================================
# HIGH-LEVEL MESSAGE FUNCTIONS (used by everyone)
# ============================================================================

def pack_log(message: str) -> bytes:
    """Create a LOG packet from a string message."""
    return pack(PacketType.LOG, message.encode('utf-8'))


def pack_error(message: str) -> bytes:
    """Create an ERROR packet from a string message."""
    return pack(PacketType.ERROR, message.encode('utf-8'))


# DEPRECATION WARNING: THIS IS A FAKE-ASS METHOD.
# AND IF YOU CATCH YOURSELF TRYING TO USE IT 
# YOU'RE ROLEPLAYING AS A C++ WEENIE 
# PLAYING AROUND WITH PRETEND BYTE OFFSETS
# INSTEAD OF WRITING WORKING JSON-RPC
def pack_data(data: bytes) -> bytes:
    """
    Create a DATA packet from raw bytes.
    
    For the teapot renderer, this contains:
    params_json + b'|||' + image_bytes
    
    This delimiter is internal to the DATA payload and not part of the protocol.
    """
    return pack(PacketType.DATA, data)

# this is how everything but the stupid pipes-only blender subprocess alternate python sandbox communicates.
def pack_jsonrpc(method: str, params: Any = None, result: Any = None, error: Any = None, id: Any = None) -> bytes:
    """
    Create a JSON-RPC packet.
    
    For requests:
        pack_jsonrpc(method="render", params={"resolution_x": 256, "resolution_y": 256}, id=1)
    
    For responses:
        pack_jsonrpc(result={"status": "ok"}, id=1)
        pack_jsonrpc(error={"code": -1, "message": "Render failed"}, id=1)
    """
    msg = {"jsonrpc": "2.0"}
    
    if method is not None:
        msg["method"] = method
        if params is not None:
            msg["params"] = params
    
    if result is not None:
        msg["result"] = result
    
    if error is not None:
        msg["error"] = error
    
    if id is not None:
        msg["id"] = id
    
    return pack(PacketType.JSONRPC, json.dumps(msg).encode('utf-8'))


def unpack_jsonrpc(payload: bytes) -> dict:
    """Parse a JSON-RPC payload into a Python dict."""
    return json.loads(payload.decode('utf-8'))


# ============================================================================
# DATA PACKET HELPERS (for the teapot use case)
# ============================================================================

def pack_teapot_data(params: dict, image_bytes: bytes) -> bytes:
    """
    Create the raw JSON/Base64 payload for teapot render results.
    
    Args:
        params: Dictionary of render parameters
        image_bytes: Raw image data as bytes
    
    Returns:
        JSON-encoded bytes of the payload
    """
    import base64
    
    payload = {
        "params": params,
        "image": base64.b64encode(image_bytes).decode('ascii')
    }
    
    # Returns the JSON bytes, NOT wrapped in the IPC header
    return json.dumps(payload).encode('utf-8')

# --- NEW FUNCTION FOR IPC PIPE USE (Used by generate_utahs.py) ---
def _pack_teapot_data_ipc(params: dict, image_bytes: bytes) -> bytes:
    """
    Create a DATA packet containing teapot render results (full TPOT header).
    This is for internal IPC use only.
    """
    payload_bytes = pack_teapot_data(params, image_bytes)
    return pack(PacketType.DATA, payload_bytes)
# -----------------------------------------------------------------

def unpack_teapot_data(payload: bytes) -> Tuple[dict, bytes]:
    """
    Parse a teapot DATA packet into params and image.
    
    Args:
        payload: The payload from a DATA packet
    
    Returns:
        (params_dict, image_bytes)
    """
    import base64
    
    data = json.loads(payload.decode('utf-8'))
    params = data["params"]
    image_bytes = base64.b64decode(data["image"])
    
    return params, image_bytes