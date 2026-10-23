"""Evidence-backed review flags (Sections 5.8, 9.4, Appendix C).

Every flag:
* names the measurable signal and the evidence values;
* states uncertainty;
* is ``requires_human_review = True``;
* compares the candidate only with their own baseline.

Script-like delivery combines several weak indicators over a sufficiently
long answer; it is never presented as proof of reading or AI authorship.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from app.analysis.relevance import RelevanceResult
from app.schemas.events import ReviewFlag


@dataclass
class FlagThresholds:
    energy_high_ratio: float = 1.6
    energy_low_ratio: float = 0.55
    pitch_change_ratio: float = 0.18  # +/- 18 % vs baseline median F0
    pace_change_ratio: float = 0.35
    long_pause_ms: int = 5000
    script_min_answer_ms: int = 20_000
    script_max_f0_cv: float = 0.08     # very flat intonation
    script_max_pause_cv: float = 0.35  # very regular pause spacing
    script_min_wpm: float = 150.0


def _fid() -> str:
    return f"flag-{uuid.uuid4().hex[:8]}"


def energy_flag(start_ms: int, end_ms: int, baseline_rms: float, segment_rms: float,
                t: FlagThresholds = FlagThresholds()) -> ReviewFlag | None:
    if baseline_rms <= 0:
        return None
    ratio = segment_rms / baseline_rms
    if t.energy_low_ratio < ratio < t.energy_high_ratio:
        return None
    direction = "increased" if ratio >= t.energy_high_ratio else "decreased"
    return ReviewFlag(
        flag_id=_fid(), category="voice_energy_change", start_ms=start_ms, end_ms=end_ms,
        explanation=f"Speech energy {direction} relative to this candidate's baseline.",
        evidence={"baseline_rms": round(baseline_rms, 4), "segment_rms": round(segment_rms, 4),
                  "ratio": round(ratio, 2)},
        uncertainty="Microphone gain, distance, and automatic gain control may affect this measurement.",
    )


def pitch_flag(start_ms: int, end_ms: int, baseline_f0: float, segment_f0: float,
               t: FlagThresholds = FlagThresholds()) -> ReviewFlag | None:
    if baseline_f0 <= 0:
        return None
    change = (segment_f0 - baseline_f0) / baseline_f0
    if abs(change) < t.pitch_change_ratio:
        return None
    direction = "higher" if change > 0 else "lower"
    return ReviewFlag(
        flag_id=_fid(), category="pitch_change", start_ms=start_ms, end_ms=end_ms,
        explanation=f"Median pitch was {direction} than this candidate's baseline.",
        evidence={"baseline_f0_hz": round(baseline_f0, 1), "segment_f0_hz": round(segment_f0, 1),
                  "relative_change": round(change, 2)},
        uncertainty="Pitch varies naturally with emphasis, emotion, and topic; pitch tracking can err on noisy audio. "
                    "A change is not evidence of stress or dishonesty.",
    )


def pace_flag(start_ms: int, end_ms: int, baseline_wpm: float, segment_wpm: float,
              t: FlagThresholds = FlagThresholds()) -> ReviewFlag | None:
    if baseline_wpm <= 0:
        return None
    change = (segment_wpm - baseline_wpm) / baseline_wpm
    if abs(change) < t.pace_change_ratio:
        return None
    direction = "faster" if change > 0 else "slower"
    return ReviewFlag(
        flag_id=_fid(), category="pace_change", start_ms=start_ms, end_ms=end_ms,
        explanation=f"Speaking rate was {direction} than this candidate's baseline.",
        evidence={"baseline_wpm": round(baseline_wpm, 1), "segment_wpm": round(segment_wpm, 1),
                  "relative_change": round(change, 2)},
        uncertainty="Rate depends on transcript word timing and topic complexity.",
    )


def long_pause_flag(start_ms: int, end_ms: int,
                    t: FlagThresholds = FlagThresholds()) -> ReviewFlag | None:
    dur = end_ms - start_ms
    if dur < t.long_pause_ms:
        return None
    return ReviewFlag(
        flag_id=_fid(), category="long_pause", start_ms=start_ms, end_ms=end_ms,
        explanation=f"Pause of {dur / 1000:.1f}s before or during the candidate's answer.",
        evidence={"pause_ms": dur},
        uncertainty="Thinking time is normal; network gaps or muting can also appear as pauses.",
    )


def relevance_flag(start_ms: int, end_ms: int, question_id: str | None,
                   result: RelevanceResult) -> ReviewFlag | None:
    if result.label not in {"off_topic", "partially_relevant"}:
        return None
    return ReviewFlag(
        flag_id=_fid(), category="low_relevance", start_ms=start_ms, end_ms=end_ms,
        explanation=f"Answer assessed as {result.label.replace('_', ' ')}: {result.rationale}",
        evidence={"question_id": question_id, "score": result.score, **result.evidence},
        uncertainty=result.uncertainty,
    )


def script_like_flag(start_ms: int, end_ms: int, f0_values: list[float],
                     pause_ms: list[int], wpm: float | None,
                     t: FlagThresholds = FlagThresholds()) -> ReviewFlag | None:
    """Combine weak indicators; require at least two to agree on a long answer."""
    if end_ms - start_ms < t.script_min_answer_ms:
        return None
    indicators: dict[str, float] = {}
    if len(f0_values) >= 8:
        arr = np.asarray(f0_values, dtype=float)
        f0_cv = float(arr.std() / arr.mean()) if arr.mean() else 1.0
        if f0_cv <= t.script_max_f0_cv:
            indicators["low_intonation_variation_cv"] = round(f0_cv, 3)
    if len(pause_ms) >= 4:
        arr = np.asarray(pause_ms, dtype=float)
        p_cv = float(arr.std() / arr.mean()) if arr.mean() else 1.0
        if p_cv <= t.script_max_pause_cv:
            indicators["regular_pause_spacing_cv"] = round(p_cv, 3)
        if arr.max() < 1200:
            indicators["no_long_hesitations_max_pause_ms"] = float(arr.max())
    if wpm is not None and wpm >= t.script_min_wpm:
        indicators["sustained_fast_rate_wpm"] = round(wpm, 1)
    if len(indicators) < 2:
        return None
    return ReviewFlag(
        flag_id=_fid(), category="script_like_delivery", start_ms=start_ms, end_ms=end_ms,
        explanation="Several delivery patterns resembled prepared or read speech. "
                    "This is a review cue only — it does not show reading or AI authorship.",
        evidence={"indicators": indicators},
        uncertainty="Rehearsed, confident, neurodivergent, or naturally steady speakers may show these "
                    "patterns. Voice alone cannot establish authorship or intent.",
    )
