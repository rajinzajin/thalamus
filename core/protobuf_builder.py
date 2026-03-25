from __future__ import annotations
"""
Cursor protobuf request body builder & x-cursor-checksum generator.

Builds the custom-framed protobuf body for Cursor's StreamUnifiedChatWithTools
RPC and generates the obfuscated x-cursor-checksum header that api2.cursor.sh
validates on every request.

Port of: cursor_protobuf_request_body_builder_and_checksum_generator.js
"""

import base64
import gzip
import hashlib
import re
import struct
import time
from datetime import datetime, timezone
from uuid import uuid4

from proto import cursor_api_pb2 as pb

from utils.structured_logging import ThalamusStructuredLogger

logger = ThalamusStructuredLogger.get_logger("protobuf-builder", "DEBUG")


# ---------------------------------------------------------------------------
# Multimodal content parsing
# ---------------------------------------------------------------------------

def parse_multimodal_content(content) -> dict:
    """Convert OpenAI multimodal content to Cursor's {content, image} format.

    OpenAI format: string OR list of {type, text/image_url}.
    Cursor format: content (str) + optional image (bytes + metadata).
    """
    if isinstance(content, str):
        return {"content": content, "image": None}

    if isinstance(content, list):
        text_parts: list[str] = []
        image_data = None

        for part in content:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:image/"):
                    m = re.match(r"^data:image/[^;]+;base64,(.+)$", url)
                    if m:
                        detail = (part.get("image_url") or {}).get("detail", "high")
                        dim = 512 if detail == "low" else 1024
                        image_data = {
                            "data": base64.b64decode(m.group(1)),
                            "metadata": {"width": dim, "height": dim},
                        }

        return {
            "content": "\n".join(text_parts) or "[User sent an image]",
            "image": image_data,
        }

    return {"content": "", "image": None}


# ---------------------------------------------------------------------------
# Protobuf body builder
# ---------------------------------------------------------------------------

def build_gzip_framed_protobuf_chat_request_body(
    messages: list[dict],
    model_name: str,
    agent_mode: bool = False,
) -> bytes:
    """Build a framed (magic + length + payload) protobuf body for Cursor API.

    Frame format: [1-byte magic] [4-byte big-endian length] [payload]
    magic 0x00 = raw protobuf, 0x01 = gzip-compressed protobuf.
    Payloads >1024 bytes are gzip-compressed.
    """
    mode_enum = 2 if agent_mode else 1
    mode_string = "Agent" if agent_mode else "Ask"

    parsed = [{**msg, **parse_multimodal_content(msg.get("content", ""))} for msg in messages]
    total_content_length = sum(len(m.get("content", "")) for m in parsed)

    instruction_text = "\n".join(
        m["content"] for m in parsed if m.get("role") == "system"
    )

    formatted: list[dict] = []

    if instruction_text:
        formatted.append({
            "content": f"[system]\n{instruction_text}\n[/system]",
            "role": 1,
            "messageId": str(uuid4()),
            "chatModeEnum": mode_enum,
        })

    for m in parsed:
        if m.get("role") == "system":
            continue
        role_int = 1 if m["role"] == "user" else 2
        msg_id = str(uuid4())
        entry: dict = {
            "content": m["content"],
            "role": role_int,
            "messageId": msg_id,
        }
        if m["role"] == "user":
            entry["chatModeEnum"] = mode_enum
        if m.get("image"):
            entry["image"] = m["image"]
        formatted.append(entry)

    message_ids = []
    for m in formatted:
        mid = {"messageId": m["messageId"], "role": m["role"]}
        if m.get("summaryId"):
            mid["summaryId"] = m["summaryId"]
        message_ids.append(mid)

    # -- Build protobuf message via the generated classes --
    req = pb.StreamUnifiedChatWithToolsRequest()
    r = req.request

    for m in formatted:
        proto_msg = r.messages.add()
        proto_msg.content = m["content"]
        proto_msg.role = m["role"]
        proto_msg.messageId = m["messageId"]
        if "chatModeEnum" in m:
            proto_msg.chatModeEnum = m["chatModeEnum"]
        if m.get("image"):
            proto_msg.image.data = m["image"]["data"]
            proto_msg.image.metadata.width = m["image"]["metadata"]["width"]
            proto_msg.image.metadata.height = m["image"]["metadata"]["height"]

    r.unknown2 = 1
    r.instruction.instruction = instruction_text
    r.unknown4 = 1
    r.model.name = model_name
    r.model.empty = b""
    r.webTool = ""
    r.unknown13 = 1

    r.cursorSetting.name = "cursor\\aisettings"
    r.cursorSetting.unknown3 = b""
    r.cursorSetting.unknown6.unknown1 = b""
    r.cursorSetting.unknown6.unknown2 = b""
    r.cursorSetting.unknown8 = 1
    r.cursorSetting.unknown9 = 1

    r.unknown19 = 1
    r.conversationId = str(uuid4())

    r.metadata.os = "win32"
    r.metadata.arch = "x64"
    r.metadata.version = "10.0.22631"
    r.metadata.path = "C:\\Program Files\\PowerShell\\7\\pwsh.exe"
    r.metadata.timestamp = datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")

    r.unknown27 = 1 if agent_mode else 0

    # supportedTools left empty — tool calling is handled entirely via
    # prompt injection + text-based JSON parsing, not Cursor's native
    # protobuf tool call mechanism.

    for mid in message_ids:
        proto_mid = r.messageIds.add()
        proto_mid.messageId = mid["messageId"]
        proto_mid.role = mid["role"]
        if "summaryId" in mid:
            proto_mid.summaryId = mid["summaryId"]

    r.largeContext = 1 if total_content_length > 20000 else 0
    r.unknown38 = 0
    r.chatModeEnum = mode_enum
    r.unknown47 = ""
    r.unknown48 = 1 if agent_mode else 0
    r.unknown49 = 0
    r.unknown51 = 0
    r.unknown53 = 1
    r.chatMode = mode_string

    payload = req.SerializeToString()

    magic = 0x00
    if len(payload) > 1024:
        payload = gzip.compress(payload)
        magic = 0x01

    return bytes([magic]) + struct.pack(">I", len(payload)) + payload


# ---------------------------------------------------------------------------
# Checksum / obfuscation helpers
# ---------------------------------------------------------------------------

def compute_sha256_hex_digest(input_str: str, salt: str = "") -> str:
    """SHA-256 hex digest of (input_str + salt)."""
    return hashlib.sha256((input_str + salt).encode()).hexdigest()


def apply_xor_chain_obfuscation(byte_array: bytearray) -> bytearray:
    """XOR-chain obfuscation extracted from the Cursor client JS bundle."""
    t = 165
    for i in range(len(byte_array)):
        byte_array[i] = ((byte_array[i] ^ t) + (i % 256)) & 0xFF
        t = byte_array[i]
    return byte_array


def generate_obfuscated_machine_id_checksum(token: str) -> str:
    """Build the x-cursor-checksum header value.

    Combines XOR-chain-obfuscated timestamp bytes with SHA-256 machine-id
    hashes derived from the auth token.
    """
    machine_id = compute_sha256_hex_digest(token, "machineId")
    mac_machine_id = compute_sha256_hex_digest(token, "macMachineId")

    timestamp = int(time.time() * 1000) // 1_000_000

    raw = bytearray([
        (timestamp >> 40) & 0xFF,
        (timestamp >> 32) & 0xFF,
        (timestamp >> 24) & 0xFF,
        (timestamp >> 16) & 0xFF,
        (timestamp >> 8) & 0xFF,
        timestamp & 0xFF,
    ])

    obfuscated = apply_xor_chain_obfuscation(raw)
    encoded = base64.b64encode(bytes(obfuscated)).decode("ascii")

    return f"{encoded}{machine_id}/{mac_machine_id}"
