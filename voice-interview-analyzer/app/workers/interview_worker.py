"""Server-side interview analysis worker (Sections 4.1, 4.4, 5).

Two layers:

``InterviewPipeline``
    Pure-Python, transport-agnostic analysis for one session. It receives
    role-labelled PCM frames and transcript results, keeps interviewer and
    candidate state separate, and emits structured dashboard events. Fully
    unit-testable without LiveKit.

``LiveKitInterviewWorker``
    Joins the LiveKit room as a hidden, subscribe-only participant, maps each
    remote audio track to a role using the server-issued identity, and feeds
    decoded frames into the pipeline. Handles mute, leave and reconnect so
    resumed tracks are never silently mislabelled (FR-19).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np

from app.analysis import review_flags as rf
from app.analysis.answer_matching import AnswerMatcher
from app.analysis.question_detection import detect_questions
from app.analysis.relevance import RelevanceScorer, get_default_scorer
from app.analysis.voice_features import (
    CandidateBaseline,
    compute_window_features,
    speech_rate_wpm,
)
from app.audio.normalization import float32_to_pcm16, normalize_frame
from app.audio.timestamps import SessionClock, TrackTimeline
from app.call.tokens import issue_worker_token, role_from_identity
from app.config import Settings
from app.schemas.events import (
    ROLES,
    AnswerAnalysis,
    AnswerAnalysisEvent,
    ErrorEvent,
    Question,
    QuestionDetectedEvent,
    ReviewFlag,
    ReviewFlagEvent,
    Role,
    SegmentCompletedEvent,
    SessionStatusEvent,
    TranscriptEvent,
    TranscriptSegment,
    VoiceMetrics,
    VoiceMetricsEvent,
)
from app.segmentation.vad import EnergyVAD
from app.transcription.service import TranscriptionStream, TranscriptResult, create_stream

log = logging.getLogger(__name__)

EmitFn = Callable[[object], Awaitable[None]]

METRIC_WINDOW_MS = 1000
FLAG_COOLDOWN_MS = 15_000


@dataclass
class RoleTrackState:
    role: Role
    sample_rate_hz: int
    timeline: TrackTimeline
    vad: EnergyVAD
    track_id: str | None = None
    stt: TranscriptionStream | None = None
    window: list[np.ndarray] = field(default_factory=list)
    window_samples: int = 0
    window_start_ms: int | None = None


@dataclass
class AnswerTurn:
    answer_id: str
    question: Question | None
    status: str
    method: str
    texts: list[str]
    start_ms: int
    end_ms: int

    @property
    def text(self) -> str:
        return " ".join(self.texts)


class InterviewPipeline:
    def __init__(self, session_id: str, emit: EmitFn, settings: Settings,
                 clock: SessionClock | None = None,
                 scorer: RelevanceScorer | None = None,
                 thresholds: rf.FlagThresholds | None = None) -> None:
        self.session_id = session_id
        self._emit = emit
        self.settings = settings
        self.clock = clock or SessionClock()
        self.scorer = scorer or get_default_scorer()
        self.t = thresholds or rf.FlagThresholds()
        self.sr = settings.analysis_sample_rate_hz
        self.matcher = AnswerMatcher()
        self.baseline = CandidateBaseline()

        self.tracks: dict[Role, RoleTrackState] = {
            role: RoleTrackState(role, self.sr, TrackTimeline(self.sr), EnergyVAD(sample_rate_hz=self.sr))
            for role in ROLES
        }
        self.transcript: list[TranscriptSegment] = []
        self.questions: list[Question] = []
        self.answers: dict[str, AnswerAnalysis] = {}
        self.metrics: list[VoiceMetrics] = []
        self.flags: list[ReviewFlag] = []
        self.f0_series: list[tuple[int, float]] = []

        self._open_answer: AnswerTurn | None = None
        self._last_flag_ms: dict[str, int] = {}
        self._stt_tasks: list[asyncio.Task] = []
        self._started = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ setup
    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for role, state in self.tracks.items():
            try:
                state.stt = create_stream(self.settings, role, self.sr)
                await state.stt.start()
                self._stt_tasks.append(asyncio.create_task(self._consume_stt(state.stt)))
            except Exception as exc:
                state.stt = None
                await self._emit(ErrorEvent(session_id=self.session_id, code="transcription_unavailable",
                                            message=f"{role}: {exc}", recoverable=True))

    async def _consume_stt(self, stream: TranscriptionStream) -> None:
        async for result in stream.results():
            await self.on_transcript(result)

    # ------------------------------------------------------------------ audio
    async def on_audio_frame(self, role: Role, pcm16: bytes, sample_rate_hz: int, channels: int,
                             server_received_ms: int, track_id: str) -> None:
        await self.start()
        state = self.tracks[role]
        if state.track_id != track_id:
            # New or republished track: re-anchor timing rather than splice.
            state.track_id = track_id
            state.timeline.reset()

        samples = normalize_frame(pcm16, sample_rate_hz, channels, self.sr)
        start_ms, _end_ms = state.timeline.place(samples.size, server_received_ms)

        for seg in state.vad.process(samples, start_ms):
            await self._emit(SegmentCompletedEvent(session_id=self.session_id, role=role,
                                                   segment_id=seg.segment_id,
                                                   start_ms=seg.start_ms, end_ms=seg.end_ms))
        if state.stt is not None:
            await state.stt.send_audio(float32_to_pcm16(samples), start_ms)
        if role == "candidate":
            await self._accumulate_window(state, samples, start_ms)

    async def _accumulate_window(self, state: RoleTrackState, samples: np.ndarray, start_ms: int) -> None:
        if state.window_start_ms is None:
            state.window_start_ms = start_ms
        state.window.append(samples)
        state.window_samples += samples.size
        if state.window_samples * 1000 / self.sr < METRIC_WINDOW_MS:
            return
        buf = np.concatenate(state.window)
        w_start = state.window_start_ms
        state.window, state.window_samples, state.window_start_ms = [], 0, None
        await self._analyse_window(buf, w_start)

    async def _analyse_window(self, buf: np.ndarray, start_ms: int) -> None:
        feats = compute_window_features(buf, self.sr, start_ms)
        vad = self.tracks["candidate"].vad
        is_speech = feats.rms > max(vad.noise_floor * vad.threshold_ratio, vad.absolute_floor)
        if not is_speech:
            return
        f0 = feats.f0_hz if feats.voiced_ratio >= 0.2 else None
        self.baseline.update(feats.rms, f0)
        if f0 is not None:
            self.f0_series.append((start_ms, f0))
        m = VoiceMetrics(
            role="candidate", start_ms=feats.start_ms, end_ms=feats.end_ms,
            rms=round(feats.rms, 5), f0_hz=round(f0, 1) if f0 else None,
            voiced_ratio=round(feats.voiced_ratio, 2),
            rms_vs_baseline=_r(self.baseline.ratio("rms", feats.rms)),
            f0_vs_baseline=_r(self.baseline.ratio("f0", f0)),
        )
        self.metrics.append(m)
        await self._emit(VoiceMetricsEvent(session_id=self.session_id, metrics=m))

        if self.baseline.ready:
            await self._maybe_flag(rf.energy_flag(m.start_ms, m.end_ms, self.baseline.rms_stat.mean,
                                                  feats.rms, self.t))
            if f0 is not None and self.baseline.f0_stat.n >= 5:
                await self._maybe_flag(rf.pitch_flag(m.start_ms, m.end_ms, self.baseline.f0_stat.mean,
                                                     f0, self.t))

    # ------------------------------------------------------------- transcript
    async def on_transcript(self, result: TranscriptResult) -> None:
        if not result.is_final:
            await self._emit(TranscriptEvent(session_id=self.session_id, role=result.role,
                                             segment_id=result.item_id or "partial",
                                             text=result.text, start_ms=result.start_ms,
                                             end_ms=result.end_ms, is_final=False))
            return
        seg = TranscriptSegment(
            segment_id=result.item_id or f"tr-{uuid.uuid4().hex[:10]}", role=result.role,
            text=result.text.strip(), start_ms=result.start_ms, end_ms=result.end_ms,
            is_final=True, confidence=result.confidence,
        )
        await self.add_final_transcript(seg)

    async def add_final_transcript(self, seg: TranscriptSegment) -> None:
        async with self._lock:
            self.transcript.append(seg)
            await self._emit(TranscriptEvent(session_id=self.session_id, role=seg.role,
                                             segment_id=seg.segment_id, text=seg.text,
                                             start_ms=seg.start_ms, end_ms=seg.end_ms, is_final=True))
            if seg.role == "interviewer":
                await self._on_interviewer_segment(seg)
            else:
                await self._on_candidate_segment(seg)

    async def _on_interviewer_segment(self, seg: TranscriptSegment) -> None:
        found = detect_questions(seg)
        if found:
            # A new question closes the candidate's previous answer turn.
            await self._finalize_answer()
        for q in found:
            self.questions.append(q)
            await self._emit(QuestionDetectedEvent(session_id=self.session_id, question=q))

    async def _on_candidate_segment(self, seg: TranscriptSegment) -> None:
        wpm = speech_rate_wpm(seg.text, seg.end_ms - seg.start_ms)
        if wpm is not None and seg.end_ms - seg.start_ms >= 3000:
            if self.baseline.rate_stat.n >= 3:
                await self._maybe_flag(rf.pace_flag(seg.start_ms, seg.end_ms,
                                                    self.baseline.rate_stat.mean, wpm, self.t))
            self.baseline.add_rate(wpm)

        match = self.matcher.match(seg, self.questions)
        qid = match.question.question_id if match.question else None
        turn = self._open_answer
        same_turn = (
            turn is not None and match.status == "paired" and turn.status == "paired"
            and turn.question is not None and turn.question.question_id == qid
        )
        if same_turn:
            assert turn is not None
            turn.texts.append(seg.text)
            turn.end_ms = seg.end_ms
        else:
            await self._finalize_answer()
            turn = AnswerTurn(f"ans-{uuid.uuid4().hex[:8]}", match.question, match.status,
                              match.method, [seg.text], seg.start_ms, seg.end_ms)
            self._open_answer = turn
            if match.question and match.status == "paired":
                await self._maybe_flag(rf.long_pause_flag(match.question.end_ms, seg.start_ms, self.t),
                                       cooldown=False)
        # Provisional analysis; updated as the turn grows (same answer_id).
        await self._emit_answer(turn, final=False)

    async def _emit_answer(self, turn: AnswerTurn, final: bool) -> AnswerAnalysis:
        rel = None
        if turn.status == "paired" and turn.question is not None:
            rel = self.scorer.score(turn.question.text, turn.text)
        a = AnswerAnalysis(
            answer_id=turn.answer_id,
            question_id=turn.question.question_id if turn.question else None,
            status=turn.status,  # type: ignore[arg-type]
            association_method=turn.method,
            answer_text=turn.text,
            start_ms=turn.start_ms,
            end_ms=turn.end_ms,
            relevance=rel.label if rel else None,  # type: ignore[arg-type]
            relevance_score=rel.score if rel else None,
            rationale=(rel.rationale if rel else _status_rationale(turn.status))
            + ("" if final else " (provisional)"),
            evidence=rel.evidence if rel else {},
            uncertainty=rel.uncertainty if rel else "Question association could not be established automatically.",
        )
        self.answers[a.answer_id] = a
        await self._emit(AnswerAnalysisEvent(session_id=self.session_id, answer=a))
        if final and rel is not None:
            await self._maybe_flag(rf.relevance_flag(turn.start_ms, turn.end_ms, a.question_id, rel),
                                   cooldown=False)
        return a

    async def _finalize_answer(self) -> None:
        turn = self._open_answer
        if turn is None:
            return
        self._open_answer = None
        await self._emit_answer(turn, final=True)
        if turn.status == "paired":
            f0_vals = [f for (t, f) in self.f0_series if turn.start_ms <= t <= turn.end_ms]
            pauses = [e - s for (s, e) in self.tracks["candidate"].vad.pauses
                      if turn.start_ms <= s and e <= turn.end_ms]
            wpm = speech_rate_wpm(turn.text, turn.end_ms - turn.start_ms)
            await self._maybe_flag(rf.script_like_flag(turn.start_ms, turn.end_ms, f0_vals, pauses, wpm, self.t),
                                   cooldown=False)

    # ------------------------------------------------------------------ flags
    async def _maybe_flag(self, flag: ReviewFlag | None, cooldown: bool = True) -> None:
        if flag is None:
            return
        if cooldown:
            last = self._last_flag_ms.get(flag.category)
            if last is not None and flag.start_ms - last < FLAG_COOLDOWN_MS:
                return
            self._last_flag_ms[flag.category] = flag.start_ms
        self.flags.append(flag)
        await self._emit(ReviewFlagEvent(session_id=self.session_id, flag=flag))

    # --------------------------------------------------------------- lifecycle
    async def on_track_ended(self, role: Role) -> None:
        state = self.tracks[role]
        for seg in state.vad.flush():
            await self._emit(SegmentCompletedEvent(session_id=self.session_id, role=role,
                                                   segment_id=seg.segment_id,
                                                   start_ms=seg.start_ms, end_ms=seg.end_ms))
        state.timeline.reset()
        state.track_id = None

    async def finish(self) -> None:
        for role in ROLES:
            await self.on_track_ended(role)
        async with self._lock:
            await self._finalize_answer()
        for state in self.tracks.values():
            if state.stt is not None:
                await state.stt.close()
        for t in self._stt_tasks:
            t.cancel()


def _r(x: float | None) -> float | None:
    return round(x, 3) if x is not None else None


def _status_rationale(status: str) -> str:
    return {
        "unresolved": "No preceding interviewer question could be associated with this answer.",
        "clarification": "Candidate appears to be asking a clarifying question.",
    }.get(status, "")


# ---------------------------------------------------------------------------
# LiveKit transport
# ---------------------------------------------------------------------------


class LiveKitInterviewWorker:  # pragma: no cover - requires a live LiveKit server
    """Hidden subscriber that feeds per-role tracks into ``InterviewPipeline``."""

    CAPTURE_RATE_HZ = 48000

    def __init__(self, settings: Settings, session_id: str, room_name: str,
                 pipeline: InterviewPipeline, emit: EmitFn) -> None:
        self.settings = settings
        self.session_id = session_id
        self.room_name = room_name
        self.pipeline = pipeline
        self._emit = emit
        self._stop = asyncio.Event()
        self._tasks: dict[str, asyncio.Task] = {}

    def stop(self) -> None:
        self._stop.set()

    async def _status(self, status: str, message: str, role: Role | None = None) -> None:
        await self._emit(SessionStatusEvent(session_id=self.session_id, status=status,
                                            message=message, role=role))

    async def run(self) -> None:
        from livekit import rtc

        room = rtc.Room()
        loop = asyncio.get_running_loop()

        def spawn(coro) -> None:
            loop.create_task(coro)

        @room.on("track_subscribed")
        def _on_track(track, publication, participant):
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            role = role_from_identity(participant.identity)
            if role is None:
                log.warning("Ignoring track from unrecognised identity %s", participant.identity)
                return
            old = self._tasks.pop(role, None)
            if old:
                old.cancel()
            self._tasks[role] = loop.create_task(self._consume(rtc, track, role, publication.sid))
            spawn(self._status("track_subscribed", f"{role} audio track connected", role))

        @room.on("track_unsubscribed")
        def _on_unsub(track, publication, participant):
            role = role_from_identity(participant.identity)
            if role:
                task = self._tasks.pop(role, None)
                if task:
                    task.cancel()
                spawn(self.pipeline.on_track_ended(role))
                spawn(self._status("track_unsubscribed", f"{role} audio track ended", role))

        @room.on("track_muted")
        def _on_muted(participant, publication):
            role = role_from_identity(participant.identity)
            if role:
                spawn(self._status("muted", f"{role} muted", role))

        @room.on("track_unmuted")
        def _on_unmuted(participant, publication):
            role = role_from_identity(participant.identity)
            if role:
                spawn(self._status("unmuted", f"{role} unmuted", role))

        @room.on("participant_connected")
        def _on_join(participant):
            role = role_from_identity(participant.identity)
            if role:
                spawn(self._status("participant_joined", f"{role} joined the call", role))

        @room.on("participant_disconnected")
        def _on_leave(participant):
            role = role_from_identity(participant.identity)
            if role:
                spawn(self._status("participant_left", f"{role} left the call", role))

        @room.on("reconnecting")
        def _on_reconnecting():
            spawn(self._status("worker_reconnecting", "Analysis worker reconnecting to media room"))

        @room.on("reconnected")
        def _on_reconnected():
            spawn(self._status("worker_reconnected", "Analysis worker reconnected"))

        token = issue_worker_token(self.settings, session_id=self.session_id, room_name=self.room_name)
        await room.connect(self.settings.livekit_url, token, rtc.RoomOptions(auto_subscribe=True))
        await self._status("worker_connected", "Analysis worker subscribed to media room")
        try:
            await self._stop.wait()
        finally:
            for t in self._tasks.values():
                t.cancel()
            await room.disconnect()

    async def _consume(self, rtc, track, role: Role, track_sid: str) -> None:
        stream = rtc.AudioStream(track, sample_rate=self.CAPTURE_RATE_HZ, num_channels=1)
        try:
            async for ev in stream:
                f = ev.frame
                await self.pipeline.on_audio_frame(
                    role, bytes(f.data), f.sample_rate, f.num_channels,
                    self.pipeline.clock.now_ms(), track_sid,
                )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.exception("Audio consumption failed for %s", role)
            await self._emit(ErrorEvent(session_id=self.session_id, code="track_processing_failed",
                                        message=f"{role}: {exc}", recoverable=True))
        finally:
            await stream.aclose()
