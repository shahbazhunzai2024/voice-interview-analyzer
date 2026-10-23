"""Live dashboard events over WebSocket (FR-18).

``WS /api/sessions/{id}/events?invite=<interviewer invite>``

Carries structured JSON events only (transcript, metrics, questions,
answers, flags, status, errors) — never call audio. On connect the server
replays the session's event history so a reconnecting dashboard catches up.
Only the interviewer role may monitor the dashboard.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from app.schemas.events import SessionStatusEvent
from app.services.session_service import NotAuthorized, SessionNotFound, session_service

router = APIRouter(tags=["events"])

PING_INTERVAL_S = 20


@router.websocket("/api/sessions/{session_id}/events")
async def session_events(websocket: WebSocket, session_id: str) -> None:
    invite = websocket.query_params.get("invite")
    try:
        session, _role = session_service.authorize(session_id, invite, require_role="interviewer")
    except (SessionNotFound, NotAuthorized):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    queue = session_service.subscribe(session)
    try:
        for payload in list(session.history):
            await websocket.send_json(payload)
        await websocket.send_json(SessionStatusEvent(
            session_id=session_id, status="dashboard_connected",
            message=f"Session is {session.status}").model_dump(mode="json"))

        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=PING_INTERVAL_S)
                await websocket.send_json(payload)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping", "session_id": session_id})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        session_service.unsubscribe(session, queue)
