"""Pydantic request/response schemas and dashboard event contracts (Section 9).

Every server event carries ``type`` and ``session_id`` so the dashboard can
route it. Times (``*_ms``) are milliseconds on the shared *session timeline*
(0 = session start on the server), not participant wall clocks.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field

Role = Literal["interviewer", "candidate"]
ROLES: tuple[Role, ...] = ("interviewer", "candidate")


def now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# REST: sessions & tokens
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    language: Literal["en"] = "en"
    roles: list[Role] = Field(default_factory=lambda: list(ROLES))
    audio_mode: Literal["in_app_webrtc"] = "in_app_webrtc"


class CreateSessionResponse(BaseModel):
    session_id: str
    status: str
    room_name: str
    # Role-specific invite codes. Each code authorizes exactly one role, so a
    # participant cannot become the interviewer by editing a client-side field.
    # Replace with real user authentication (SSO/OAuth) before production.
    invites: dict[Role, str]


class SessionInfo(BaseModel):
    session_id: str
    status: str
    language: str
    room_name: str
    created_at_ms: int
    finished_at_ms: Optional[int] = None
    connected_roles: list[Role] = Field(default_factory=list)


class JoinTokenRequest(BaseModel):
    invite_code: str = Field(min_length=8)
    display_name: Optional[str] = Field(default=None, max_length=64)


class JoinTokenResponse(BaseModel):
    token: str
    livekit_url: str
    room_name: str
    role: Role
    identity: str
    expires_in_s: int


class DevTranscriptRequest(BaseModel):
    """Development helper: inject a final transcript segment without STT."""

    role: Role
    text: str = Field(min_length=1)
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None


# ---------------------------------------------------------------------------
# Media track identity (9.2)
# ---------------------------------------------------------------------------


class TrackMetadata(BaseModel):
    source: Literal["participant_track"] = "participant_track"
    session_id: str
    role: Role
    track_id: str
    server_received_ms: int
    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(ge=1, le=8)
    encoding: Literal["pcm_s16le", "pcm_f32le"] = "pcm_s16le"


# ---------------------------------------------------------------------------
# Domain results (also embedded in events and the report)
# ---------------------------------------------------------------------------


class TranscriptSegment(BaseModel):
    segment_id: str
    role: Role
    text: str
    start_ms: int
    end_ms: int
    is_final: bool = True
    confidence: Optional[float] = None


class Question(BaseModel):
    question_id: str
    text: str
    start_ms: int
    end_ms: int
    confidence: float
    kind: Literal["direct", "indirect", "compound"] = "direct"
    source_segment_id: Optional[str] = None


class AnswerAnalysis(BaseModel):
    answer_id: str
    question_id: Optional[str]  # None => unresolved
    status: Literal["paired", "unresolved", "clarification"]
    association_method: str
    answer_text: str
    start_ms: int
    end_ms: int
    relevance: Optional[Literal["relevant", "partially_relevant", "off_topic", "insufficient_context"]] = None
    relevance_score: Optional[float] = None
    rationale: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    uncertainty: str = ""


class VoiceMetrics(BaseModel):
    role: Role
    start_ms: int
    end_ms: int
    rms: Optional[float] = None
    f0_hz: Optional[float] = None
    speech_rate_wpm: Optional[float] = None
    voiced_ratio: Optional[float] = None
    # Relative to this speaker's own baseline (1.0 == baseline).
    rms_vs_baseline: Optional[float] = None
    f0_vs_baseline: Optional[float] = None


class ReviewFlag(BaseModel):
    flag_id: str
    category: Literal[
        "voice_energy_change",
        "pitch_change",
        "pace_change",
        "long_pause",
        "low_relevance",
        "script_like_delivery",
    ]
    role: Role = "candidate"
    start_ms: int
    end_ms: int
    explanation: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    uncertainty: str
    requires_human_review: Literal[True] = True


# ---------------------------------------------------------------------------
# Dashboard events (9.3)
# ---------------------------------------------------------------------------


class _Event(BaseModel):
    session_id: str
    emitted_at_ms: int = Field(default_factory=now_ms)


class SessionStatusEvent(_Event):
    type: Literal["session_status"] = "session_status"
    status: str
    message: str = ""
    role: Optional[Role] = None


class TranscriptEvent(_Event):
    type: Literal["transcript.partial"] = "transcript.partial"
    role: Role
    segment_id: str
    text: str
    start_ms: int
    end_ms: int
    is_final: bool


class SegmentCompletedEvent(_Event):
    type: Literal["segment.completed"] = "segment.completed"
    role: Role
    segment_id: str
    start_ms: int
    end_ms: int


class VoiceMetricsEvent(_Event):
    type: Literal["voice_metrics"] = "voice_metrics"
    metrics: VoiceMetrics


class QuestionDetectedEvent(_Event):
    type: Literal["question.detected"] = "question.detected"
    question: Question


class AnswerAnalysisEvent(_Event):
    type: Literal["answer.analysis"] = "answer.analysis"
    answer: AnswerAnalysis


class ReviewFlagEvent(_Event):
    type: Literal["review.flag"] = "review.flag"
    flag: ReviewFlag


class ErrorEvent(_Event):
    type: Literal["error"] = "error"
    code: str
    message: str
    recoverable: bool = True


ServerEvent = Annotated[
    Union[
        SessionStatusEvent,
        TranscriptEvent,
        SegmentCompletedEvent,
        VoiceMetricsEvent,
        QuestionDetectedEvent,
        AnswerAnalysisEvent,
        ReviewFlagEvent,
        ErrorEvent,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Report (FR-14)
# ---------------------------------------------------------------------------


class VoiceSummary(BaseModel):
    windows: int
    baseline_rms: Optional[float] = None
    baseline_f0_hz: Optional[float] = None
    mean_speech_rate_wpm: Optional[float] = None


class InterviewReport(BaseModel):
    session_id: str
    status: str
    language: str
    generated_at_ms: int = Field(default_factory=now_ms)
    duration_ms: Optional[int] = None
    transcript: list[TranscriptSegment]
    questions: list[Question]
    answers: list[AnswerAnalysis]
    candidate_voice: VoiceSummary
    flags: list[ReviewFlag]
    limitations: list[str]
    disclaimer: str
