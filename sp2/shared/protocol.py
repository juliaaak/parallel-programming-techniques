"""
Shared protocol definitions for chat client/server.
All messages are JSON-encoded and length-prefixed (4 bytes, big-endian).
"""

import json
import struct

# ── network constants ──────────────────────────────────────────────────────────
HOST        = "0.0.0.0"   # server listens on all interfaces
CLIENT_HOST = "127.0.0.1" # default host clients connect to
PORT        = 9090
HEADER_SIZE     = 4                    # bytes for message length prefix
BUFFER_SIZE     = 4096                 # recv buffer size
MAX_FILE_SIZE   = 50 * 1024 * 1024    # 50 MB
MAX_FRAME_SIZE  = MAX_FILE_SIZE + 4096 # max raw frame (file + metadata overhead)


# ── message types ──────────────────────────────────────────────────────────────
class MsgType:
    # client → server
    REGISTER    = "register"     # {"type":"register","username":"..."}
    BROADCAST   = "broadcast"    # {"type":"broadcast","text":"...","fmt":"plain|md"}
    PRIVATE     = "private"      # {"type":"private","to":"...","text":"...","fmt":"..."}
    GROUP_MSG   = "group_msg"    # {"type":"group_msg","group":"...","text":"...","fmt":"..."}
    JOIN_GROUP  = "join_group"   # {"type":"join_group","group":"..."}
    LEAVE_GROUP = "leave_group"  # {"type":"leave_group","group":"..."}
    LIST_USERS  = "list_users"
    LIST_GROUPS = "list_groups"
    FILE_SEND   = "file_send"    # {"type":"file_send","to":"...","filename":"...","data":"<b64>","media_type":"..."}
    FILE_GROUP  = "file_group"   # {"type":"file_group","group":"...","filename":"...","data":"<b64>","media_type":"..."}
    DISCONNECT  = "disconnect"

    # server → client
    SERVER_MSG        = "server_msg"       # info/error from server
    CHAT_MSG          = "chat_msg"         # delivered chat message
    USER_LIST         = "user_list"        # {"type":"user_list","users":[...]}
    GROUP_LIST        = "group_list"       # {"type":"group_list","groups":{...}}
    FILE_RECV         = "file_recv"        # file delivery
    OFFLINE_DELIVERED = "offline_delivered"# bulk offline packet


# ── media type helpers ─────────────────────────────────────────────────────────
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"}
AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

def guess_media_type(filename: str) -> str:
    """Return MIME-like media_type based on file extension."""
    import os
    ext = os.path.splitext(filename)[1].lower()
    if ext in IMAGE_EXTS:
        mapping = {".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",
                   ".gif":"image/gif",".bmp":"image/bmp",".webp":"image/webp",
                   ".svg":"image/svg+xml"}
        return mapping.get(ext, "image/unknown")
    if ext in AUDIO_EXTS:
        return f"audio/{ext.lstrip('.')}"
    if ext in VIDEO_EXTS:
        return f"video/{ext.lstrip('.')}"
    return "application/octet-stream"


# ── wire encoding ──────────────────────────────────────────────────────────────
def encode(payload: dict) -> bytes:
    """Serialize dict → length-prefixed bytes."""
    data   = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = struct.pack(">I", len(data))
    return header + data


def decode_header(raw: bytes) -> int:
    """Extract message length from 4-byte big-endian header."""
    return struct.unpack(">I", raw)[0]