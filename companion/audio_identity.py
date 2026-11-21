"""Stable conversation scope for successive authenticated audio connections."""

import uuid


def device_audio_conversation_id(device_id: str) -> str:
    if not isinstance(device_id, str) or not 1 <= len(device_id) <= 64:
        raise ValueError("Invalid device ID")
    # The WebSocket itself gets a fresh trace ID. This separate key lets the
    # same authenticated physical Puppy finish a short rename confirmation on
    # its next wake, while Store still separates members and devices.
    return "device-audio-" + uuid.uuid5(uuid.NAMESPACE_URL, "puppy-audio:" + device_id).hex
