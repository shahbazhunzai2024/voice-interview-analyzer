"""Role-specific, short-lived LiveKit access tokens (FR-02, FR-17).

The role is decided server-side from an authorized invite and is embedded in
the token identity and metadata. Participants may only publish their
microphone; the analysis worker is hidden and subscribe-only.
"""

from __future__ import annotations

import json
from datetime import timedelta

from app.config import Settings
from app.schemas.events import Role


class LiveKitNotConfigured(RuntimeError):
    pass


def participant_identity(role: Role, session_id: str) -> str:
    # Identity encodes the role; the worker trusts it because only the
    # server can sign tokens.
    return f"{role}:{session_id}"


def role_from_identity(identity: str) -> Role | None:
    prefix = identity.split(":", 1)[0]
    return prefix if prefix in ("interviewer", "candidate") else None  # type: ignore[return-value]


def issue_join_token(settings: Settings, *, session_id: str, room_name: str, role: Role,
                     display_name: str | None = None) -> tuple[str, str]:
    """Return (jwt, identity) for a participant."""
    if not settings.livekit_configured:
        raise LiveKitNotConfigured("LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not set")
    from livekit import api

    identity = participant_identity(role, session_id)
    grants = api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=False,
        can_publish_sources=["microphone"],  # audio-only MVP
        can_update_own_metadata=False,
    )
    token = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_name(display_name or role.capitalize())
        .with_metadata(json.dumps({"role": role, "session_id": session_id}))
        .with_ttl(timedelta(seconds=settings.join_token_ttl_s))
        .with_grants(grants)
        .to_jwt()
    )
    return token, identity


def issue_worker_token(settings: Settings, *, session_id: str, room_name: str) -> str:
    """Hidden, subscribe-only token for the server-side analysis worker."""
    if not settings.livekit_configured:
        raise LiveKitNotConfigured("LiveKit is not configured")
    from livekit import api

    grants = api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=False,
        can_subscribe=True,
        can_publish_data=False,
        hidden=True,
        agent=True,
    )
    return (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(f"analysis-worker:{session_id}")
        .with_name("Analysis worker")
        .with_ttl(timedelta(hours=4))
        .with_grants(grants)
        .to_jwt()
    )
