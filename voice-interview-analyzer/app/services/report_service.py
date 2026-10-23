"""Final interview review report (FR-14).

The report summarises segments, questions, answers, voice measurements and
flags with evidence. It never produces pass/fail or ranking decisions.
"""

from __future__ import annotations

from statistics import mean

from app.schemas.events import InterviewReport, VoiceSummary
from app.services.session_service import InterviewSession

LIMITATIONS = [
    "Flags are review cues for a human reviewer, not conclusions about stress, honesty, ability, or intent.",
    "Voice alone cannot establish whether an answer was written by an AI tool or read from a script.",
    "Energy and pitch are compared only with this candidate's own baseline; microphone gain, distance, "
    "automatic gain control and noise suppression can influence them.",
    "Transcription errors can affect question detection, answer pairing and relevance scores.",
    "Accents, speech impairments, neurodiversity, culture and room acoustics can influence measured signals.",
    "Question-answer pairing may be ambiguous for follow-up or compound questions; unresolved cases are kept.",
    "Timing across the two participants is aligned by server receive time; it is not sample-accurate.",
]

DISCLAIMER = (
    "This report supports human review only. It contains no automatic pass/fail, score, or ranking "
    "and must not be used as the sole basis for a hiring decision."
)


def build_report(session: InterviewSession) -> InterviewReport:
    p = session.pipeline
    rates = [m.speech_rate_wpm for m in p.metrics if m.speech_rate_wpm is not None]
    baseline = p.baseline
    duration = None
    if session.finished_at_ms:
        duration = session.finished_at_ms - session.created_at_ms
    elif p.transcript:
        duration = max(s.end_ms for s in p.transcript)

    return InterviewReport(
        session_id=session.session_id,
        status=session.status,
        language=session.language,
        duration_ms=duration,
        transcript=sorted(p.transcript, key=lambda s: s.start_ms),
        questions=p.questions,
        answers=sorted(p.answers.values(), key=lambda a: a.start_ms),
        candidate_voice=VoiceSummary(
            windows=len(p.metrics),
            baseline_rms=round(baseline.rms_stat.mean, 5) if baseline.rms_stat.n else None,
            baseline_f0_hz=round(baseline.f0_stat.mean, 1) if baseline.f0_stat.n else None,
            mean_speech_rate_wpm=round(mean(rates), 1) if rates else (
                round(baseline.rate_stat.mean, 1) if baseline.rate_stat.n else None),
        ),
        flags=sorted(p.flags, key=lambda f: f.start_ms),
        limitations=LIMITATIONS,
        disclaimer=DISCLAIMER,
    )
