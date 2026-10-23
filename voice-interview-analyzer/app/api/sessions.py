"""Interview session REST APIs (Section 7.5 / 9.1).

Authorization model (prototype): creating a session returns one invite code
per role. A participant presents their code in the ``X-Invite-Code`` header
(or request body for join-token). The server derives the role from the code,
so changing a client-side role field cannot grant another role (FR-02).
Replace invite codes with real user authentication before production.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Header, HTTPException, status

from app.call.tokens import LiveKitNotConfigured, issue_join_token
from app.schemas.events import (
    CreateSessionRequest,
    CreateSessionResponse,
    DevTranscriptRequest,
    JoinTokenRequest,
    JoinTokenResponse,
    SessionInfo,
    TranscriptSegment,
)
from app.services.session_service import (
    InterviewSession,
    NotAuthorized,
    SessionFinished,
    SessionNotFound,
    session_service,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _authorize(session_id: str, invite: str | None, require_role=None):
    try:
        return session_service.authorize(session_id, invite, require_role)
    except SessionNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    except NotAuthorized as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from None


def _info(s: InterviewSession) -> SessionInfo:
    return SessionInfo(
        session_id=s.session_id, status=s.status, language=s.language, room_name=s.room_name,
        created_at_ms=s.created_at_ms, finished_at_ms=s.finished_at_ms,
        connected_roles=sorted(s.joined_roles),
    )


@router.post("", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(body: CreateSessionRequest | None = None) -> CreateSessionResponse:
    body = body or CreateSessionRequest()
    try:
        s = await session_service.create(language=body.language, roles=body.roles)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    return CreateSessionResponse(session_id=s.session_id, status=s.status,
                                 room_name=s.room_name, invites=s.invites)


@router.get("/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str, x_invite_code: str | None = Header(default=None)) -> SessionInfo:
    s, _role = _authorize(session_id, x_invite_code)
    return _info(s)


@router.post("/{session_id}/join-token", response_model=JoinTokenResponse)
async def join_token(session_id: str, body: JoinTokenRequest) -> JoinTokenResponse:
    s, role = _authorize(session_id, body.invite_code)
    if s.status == "finished":
        raise HTTPException(status.HTTP_409_CONFLICT, "Session has ended")
    try:
        token, identity = issue_join_token(
            session_service.settings, session_id=s.session_id, room_name=s.room_name,
            role=role, display_name=body.display_name,
        )
    except LiveKitNotConfigured as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None
    try:
        await session_service.mark_joined(s, role)
    except SessionFinished:
        raise HTTPException(status.HTTP_409_CONFLICT, "Session has ended") from None
    return JoinTokenResponse(
        token=token, livekit_url=session_service.settings.livekit_url, room_name=s.room_name,
        role=role, identity=identity, expires_in_s=session_service.settings.join_token_ttl_s,
    )


@router.post("/{session_id}/finish", response_model=SessionInfo)
async def finish_session(session_id: str, x_invite_code: str | None = Header(default=None)) -> SessionInfo:
    s, _ = _authorize(session_id, x_invite_code, require_role="interviewer")
    await session_service.finish(s)
    return _info(s)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, x_invite_code: str | None = Header(default=None)) -> None:
    """Delete transcript, metrics, flags and report data as one set."""
    s, _ = _authorize(session_id, x_invite_code, require_role="interviewer")
    await session_service.finish(s)
    session_service.delete(session_id)


@router.post("/{session_id}/dev/transcript", response_model=TranscriptSegment, tags=["dev"])
async def dev_transcript(session_id: str, body: DevTranscriptRequest,
                         x_invite_code: str | None = Header(default=None)) -> TranscriptSegment:
    """DEV_MODE only: inject a final transcript segment to exercise analysis without STT."""
    if not session_service.settings.dev_mode:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    s, _ = _authorize(session_id, x_invite_code, require_role="interviewer")
    if s.status == "finished":
        raise HTTPException(status.HTTP_409_CONFLICT, "Session has ended")
    now = s.clock.now_ms()
    words = len(body.text.split())
    end = body.end_ms if body.end_ms is not None else now
    start = body.start_ms if body.start_ms is not None else max(0, end - words * 400)
    if end < start:
        raise HTTPException(422, "end_ms must be >= start_ms")
    seg = TranscriptSegment(segment_id=f"dev-{uuid.uuid4().hex[:8]}", role=body.role,
                            text=body.text, start_ms=start, end_ms=end)
    await s.pipeline.add_final_transcript(seg)
    return seg
