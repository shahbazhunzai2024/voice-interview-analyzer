"""Interview session lifecycle (in-memory for the prototype; Section 6 Storage).

Responsibilities
* create sessions with role-specific invite codes (stand-in for real auth);
* authorize a role from an invite — never from a client-supplied role field;
* own the per-session ``InterviewPipeline`` and optional LiveKit worker;
* fan out structured events to dashboard WebSocket subscribers;
* finish sessions and release resources.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import secrets
import uuid
from dataclasses import dataclass, field

from app.audio.timestamps import SessionClock
from app.call.rooms import create_room, delete_room, room_name_for
from app.config import Settings, settings as default_settings
from app.schemas.events import ROLES, Role, SessionStatusEvent, now_ms
from app.workers.interview_worker import InterviewPipeline, LiveKitInterviewWorker

log = logging.getLogger(__name__)

EVENT_HISTORY_LIMIT = 5000


class SessionNotFound(KeyError):
    pass


class NotAuthorized(PermissionError):
    pass


class SessionFinished(RuntimeError):
    pass


@dataclass
class InterviewSession:
    session_id: str
    language: str
    room_name: str
    invites: dict[Role, str]
    pipeline: InterviewPipeline
    clock: SessionClock
    status: str = "created"
    created_at_ms: int = field(default_factory=now_ms)
    finished_at_ms: int | None = None
    joined_roles: set[Role] = field(default_factory=set)
    history: list[dict] = field(default_factory=list)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    worker: LiveKitInterviewWorker | None = None
    worker_task: asyncio.Task | None = None


class SessionService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self._sessions: dict[str, InterviewSession] = {}

    # ----------------------------------------------------------- lifecycle
    async def create(self, language: str = "en", roles: list[Role] | None = None) -> InterviewSession:
        roles = roles or list(ROLES)
        if set(roles) != set(ROLES):
            raise ValueError("A session requires exactly the interviewer and candidate roles")
        session_id = uuid.uuid4().hex[:12]
        clock = SessionClock()

        async def emit(event) -> None:
            await self.publish(session_id, event)

        session = InterviewSession(
            session_id=session_id,
            language=language,
            room_name=room_name_for(session_id),
            invites={role: secrets.token_urlsafe(18) for role in roles},
            pipeline=InterviewPipeline(session_id, emit, self.settings, clock=clock),
            clock=clock,
        )
        self._sessions[session_id] = session
        await create_room(self.settings, session.room_name)
        return session

    def get(self, session_id: str) -> InterviewSession:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFound(session_id) from None

    def authorize(self, session_id: str, invite_code: str | None,
                  require_role: Role | None = None) -> tuple[InterviewSession, Role]:
        """Resolve the role for an invite code (constant-time comparison)."""
        session = self.get(session_id)
        if not invite_code:
            raise NotAuthorized("missing invite code")
        for role, code in session.invites.items():
            if hmac.compare_digest(code.encode(), invite_code.encode()):
                if require_role and role != require_role:
                    raise NotAuthorized(f"this action requires the {require_role} role")
                return session, role
        raise NotAuthorized("invalid invite code")

    async def mark_joined(self, session: InterviewSession, role: Role) -> None:
        if session.status == "finished":
            raise SessionFinished(session.session_id)
        session.joined_roles.add(role)
        if session.status == "created":
            session.status = "active"
        await self.publish(session.session_id, SessionStatusEvent(
            session_id=session.session_id, status="join_authorized",
            message=f"{role} was issued a join token", role=role))
        self._ensure_worker(session)

    def _ensure_worker(self, session: InterviewSession) -> None:
        if (session.worker_task is not None or not self.settings.livekit_configured
                or not self.settings.analysis_worker_enabled):
            return

        async def emit(event) -> None:
            await self.publish(session.session_id, event)

        session.worker = LiveKitInterviewWorker(self.settings, session.session_id, session.room_name,
                                                session.pipeline, emit)
        session.worker_task = asyncio.create_task(self._run_worker(session))

    async def _run_worker(self, session: InterviewSession) -> None:  # pragma: no cover - network
        try:
            assert session.worker is not None
            await session.worker.run()
        except Exception as exc:
            log.exception("Analysis worker failed for %s", session.session_id)
            await self.publish(session.session_id, SessionStatusEvent(
                session_id=session.session_id, status="worker_error",
                message=f"Analysis worker stopped: {exc}"))

    async def finish(self, session: InterviewSession) -> None:
        if session.status == "finished":
            return
        if session.worker is not None:
            session.worker.stop()
        if session.worker_task is not None:
            try:
                await asyncio.wait_for(session.worker_task, timeout=5)
            except (asyncio.TimeoutError, Exception):
                session.worker_task.cancel()
        await session.pipeline.finish()
        await delete_room(self.settings, session.room_name)
        session.status = "finished"
        session.finished_at_ms = now_ms()
        await self.publish(session.session_id, SessionStatusEvent(
            session_id=session.session_id, status="finished",
            message="Session ended; media released and analysis finalized."))

    def delete(self, session_id: str) -> None:
        """Delete all session data as a coherent set (Section 11 retention)."""
        self._sessions.pop(session_id, None)

    # --------------------------------------------------------------- events
    async def publish(self, session_id: str, event) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        payload = event.model_dump(mode="json")
        session.history.append(payload)
        if len(session.history) > EVENT_HISTORY_LIMIT:
            del session.history[: len(session.history) - EVENT_HISTORY_LIMIT]
        for q in list(session.subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                log.warning("Dropping event for slow dashboard subscriber")

    def subscribe(self, session: InterviewSession) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        session.subscribers.add(q)
        return q

    def unsubscribe(self, session: InterviewSession, q: asyncio.Queue) -> None:
        session.subscribers.discard(q)


session_service = SessionService()
