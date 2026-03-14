# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Enrico Santagati

"""Plugin for nRF52840 — read FICR Device ID."""

import struct
from mcp.types import Tool

from dbgprobe_mcp_server.helpers import _ok, _err
from dbgprobe_mcp_server.state import ProbeState

META = {
    "description": "nRF52840 plugin — read device ID from FICR",
    "device_name_contains": "nRF52840",
}

TOOLS = [
    Tool(
        name="nrf52840.device_id",
        description="Read the 64-bit unique device ID from FICR.DEVICEID[0] and DEVICEID[1].",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    ),
]


async def handle_device_id(state: ProbeState, args: dict) -> dict:
    session = state.get_session(args["session_id"])
    backend = session.backend

    # FICR.DEVICEID[0] at 0x10000060, DEVICEID[1] at 0x10000064
    data = await backend.mem_read(0x10000060, 8)
    id0, id1 = struct.unpack("<II", data)
    device_id = (id1 << 32) | id0

    return _ok(
        device_id=f"0x{device_id:016X}",
        deviceid_0=f"0x{id0:08X}",
        deviceid_1=f"0x{id1:08X}",
    )


HANDLERS = {
    "nrf52840.device_id": handle_device_id,
}
