"""LiveKit room creation and management (media plane, Section 4.4).

LiveKit auto-creates a room when the first participant joins, so explicit
creation is optional; we still create it to set an empty-timeout and a
participant cap (interviewer + candidate + hidden analysis worker).
"""

from __future__ import annotations

import logging

from app.config import Settings

log = logging.getLogger(__name__)

MAX_PARTICIPANTS = 3  # interviewer, candidate, analysis worker


def room_name_for(session_id: str) -> str:
    return f"interview-{session_id}"


async def create_room(settings: Settings, room_name: str) -> bool:
    """Create the LiveKit room. Returns False if LiveKit is not configured."""
    if not settings.livekit_configured:
        return False
    from livekit import api  # imported lazily so the app runs without LiveKit

    lk = api.LiveKitAPI(settings.livekit_url, settings.livekit_api_key, settings.livekit_api_secret)
    try:
        await lk.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                empty_timeout=10 * 60,
                max_participants=MAX_PARTICIPANTS,
            )
        )
        return True
    except Exception as exc:  # network / auth errors surface to the caller's log
        log.warning("LiveKit create_room failed for %s: %s", room_name, exc)
        return False
    finally:
        await lk.aclose()


async def delete_room(settings: Settings, room_name: str) -> None:
    """Delete the room, disconnecting everyone (FR-13)."""
    if not settings.livekit_configured:
        return
    from livekit import api

    lk = api.LiveKitAPI(settings.livekit_url, settings.livekit_api_key, settings.livekit_api_secret)
    try:
        await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))
    except Exception as exc:
        log.warning("LiveKit delete_room failed for %s: %s", room_name, exc)
    finally:
        await lk.aclose()
