"""Final report API (FR-14). Interviewer/reviewer access only."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, status

from app.schemas.events import InterviewReport
from app.services.report_service import build_report
from app.services.session_service import NotAuthorized, SessionNotFound, session_service

router = APIRouter(prefix="/api/sessions", tags=["reports"])


@router.get("/{session_id}/report", response_model=InterviewReport)
async def get_report(session_id: str, x_invite_code: str | None = Header(default=None)) -> InterviewReport:
    try:
        session, _ = session_service.authorize(session_id, x_invite_code, require_role="interviewer")
    except SessionNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from None
    except NotAuthorized as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from None
    return build_report(session)
