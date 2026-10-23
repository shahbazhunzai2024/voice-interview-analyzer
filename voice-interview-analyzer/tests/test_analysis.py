import asyncio

import numpy as np
import pytest

from app.analysis import review_flags as rf
from app.analysis.answer_matching import AnswerMatcher
from app.analysis.question_detection import classify_sentence, detect_questions
from app.analysis.relevance import LexicalRelevanceScorer
from app.analysis.voice_features import (
    CandidateBaseline,
    estimate_f0,
    estimate_f0_autocorr,
    rms,
    speech_rate_wpm,
)
from app.audio.normalization import float32_to_pcm16
from app.config import settings
from app.schemas.events import Question, TranscriptSegment
from app.segmentation.vad import segment_audio
from app.workers.interview_worker import InterviewPipeline

SR = 16000


def tone(freq, seconds, amp=0.3, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds, sr=SR, noise=0.0005):
    rng = np.random.default_rng(0)
    return (noise * rng.standard_normal(int(sr * seconds))).astype(np.float32)


def seg(role, text, start, end, sid="s"):
    return TranscriptSegment(segment_id=sid, role=role, text=text, start_ms=start, end_ms=end)


# --------------------------------------------------------------- voice features
def test_rms_of_sine():
    assert rms(tone(200, 1, amp=0.5)) == pytest.approx(0.5 / np.sqrt(2), rel=1e-3)
    assert rms(np.zeros(0)) == 0.0


@pytest.mark.parametrize("freq", [110, 180, 240])
def test_f0_autocorr_estimates_pitch(freq):
    est = estimate_f0_autocorr(tone(freq, 0.05), SR)
    assert est == pytest.approx(freq, rel=0.03)


def test_f0_none_for_silence():
    assert estimate_f0_autocorr(silence(0.05), SR) is None
    f0, voiced = estimate_f0(silence(0.5), SR)
    assert f0 is None and voiced == 0.0


def test_speech_rate():
    assert speech_rate_wpm("one two three four", 2000) == pytest.approx(120)
    assert speech_rate_wpm("", 1000) is None


def test_baseline_freezes_after_warmup():
    b = CandidateBaseline(warmup_windows=3)
    for v in (0.1, 0.1, 0.1, 0.9):
        b.update(v, 150.0)
    assert b.ready and b.rms_stat.mean == pytest.approx(0.1)
    assert b.ratio("rms", 0.2) == pytest.approx(2.0)


# --------------------------------------------------------------------------- VAD
def test_vad_finds_speech_segments():
    audio = np.concatenate([silence(1), tone(200, 1.0), silence(1.2), tone(200, 0.8), silence(1)])
    segs = segment_audio(audio, SR)
    assert len(segs) == 2
    assert segs[0].start_ms == pytest.approx(1000, abs=60)
    assert segs[0].end_ms == pytest.approx(2000, abs=60)
    assert segs[1].start_ms == pytest.approx(3200, abs=60)


def test_vad_short_pause_does_not_split_turn():
    audio = np.concatenate([silence(0.5), tone(200, 1), silence(0.3), tone(200, 1), silence(1)])
    assert len(segment_audio(audio, SR)) == 1


def test_vad_ignores_blips():
    audio = np.concatenate([silence(1), tone(200, 0.09), silence(1)])
    assert segment_audio(audio, SR) == []


# ------------------------------------------------------------ question detection
@pytest.mark.parametrize("text,expected", [
    ("What motivated you to apply?", True),
    ("Tell me about a time you handled conflict.", True),
    ("Walk me through your last project", True),
    ("how did you scale the service", True),  # STT dropped the '?'
    ("Great, thanks.", False),
    ("That's why we use Kubernetes.", False),
    ("I worked at a bank for three years.", False),
])
def test_classify_sentence(text, expected):
    assert classify_sentence(text)[0] is expected


def test_compound_question_merged():
    qs = detect_questions(seg("interviewer", "Why did you leave? And what are you looking for next?", 0, 5000))
    assert len(qs) == 1 and qs[0].kind == "compound"


def test_candidate_segments_are_not_questions():
    assert detect_questions(seg("candidate", "What do you mean?", 0, 1000)) == []


# --------------------------------------------------------------- answer matching
def _q(qid, start, end, text="What is your experience with Python?"):
    return Question(question_id=qid, text=text, start_ms=start, end_ms=end, confidence=0.9)


def test_matches_most_recent_preceding_question():
    m = AnswerMatcher().match(seg("candidate", "I used it daily.", 12000, 15000),
                              [_q("q1", 0, 3000), _q("q2", 9000, 11000)])
    assert m.status == "paired" and m.question.question_id == "q2"


def test_unresolved_without_question():
    m = AnswerMatcher().match(seg("candidate", "Hello there.", 0, 1000), [])
    assert m.status == "unresolved" and m.question is None


def test_unresolved_when_gap_too_large():
    m = AnswerMatcher(max_gap_ms=10_000).match(seg("candidate", "Answer", 60_000, 62_000), [_q("q1", 0, 2000)])
    assert m.status == "unresolved"


def test_clarifying_question_detected():
    m = AnswerMatcher().match(seg("candidate", "Do you mean in production or in testing?", 4000, 6000),
                              [_q("q1", 0, 3000)])
    assert m.status == "clarification"


# --------------------------------------------------------------------- relevance
def test_relevance_labels():
    s = LexicalRelevanceScorer()
    q = "Tell me about your experience designing database schemas."
    assert s.score(q, "I designed database schemas for an inventory system and normalised the tables.").label == "relevant"
    off = s.score(q, "My favourite hobby is hiking in the mountains with friends on weekends.")
    assert off.label == "off_topic" and off.rationale and off.uncertainty
    assert s.score(q, "Yes.").label == "insufficient_context"


# ------------------------------------------------------------------------- flags
def test_energy_flag_matches_documented_example():
    f = rf.energy_flag(261000, 269000, baseline_rms=0.03, segment_rms=0.05)
    assert f is not None and f.category == "voice_energy_change"
    assert f.requires_human_review is True
    assert f.evidence["baseline_rms"] == 0.03 and f.evidence["segment_rms"] == 0.05
    assert "Microphone" in f.uncertainty
    assert rf.energy_flag(0, 1, 0.03, 0.035) is None


def test_script_like_requires_multiple_indicators():
    flat_f0 = [150.0 + (i % 2) for i in range(20)]
    regular_pauses = [400, 420, 410, 405, 415]
    f = rf.script_like_flag(0, 30_000, flat_f0, regular_pauses, wpm=165)
    assert f is not None and "not show reading or AI authorship" in f.explanation
    # one indicator alone is not enough; short answers are never flagged
    assert rf.script_like_flag(0, 30_000, flat_f0, [], wpm=100) is None
    assert rf.script_like_flag(0, 5_000, flat_f0, regular_pauses, wpm=165) is None


def test_long_pause_flag():
    assert rf.long_pause_flag(0, 6000) is not None
    assert rf.long_pause_flag(0, 2000) is None


# ---------------------------------------------------------------- full pipeline
def test_pipeline_keeps_roles_separate_and_flags_energy():
    events = []

    async def emit(e):
        events.append(e)

    async def scenario():
        p = InterviewPipeline("sess", emit, settings)
        frame = int(0.02 * 48000)
        t = 0
        # 12 s of steady candidate speech at 48 kHz to build the baseline, then 3 s louder
        for amp, secs in ((0.05, 12), (0.25, 3)):
            sig = tone(180, secs, amp=amp, sr=48000)
            for i in range(0, sig.size, frame):
                t += 20
                await p.on_audio_frame("candidate", float32_to_pcm16(sig[i:i + frame]), 48000, 1, t, "TR_c")
        # interviewer audio must not produce candidate metrics
        sig = tone(120, 2, amp=0.3, sr=48000)
        for i in range(0, sig.size, frame):
            await p.on_audio_frame("interviewer", float32_to_pcm16(sig[i:i + frame]), 48000, 1, t + i // 48, "TR_i")
        await p.finish()
        return p

    p = asyncio.run(scenario())
    assert p.metrics and all(m.role == "candidate" for m in p.metrics)
    assert p.baseline.ready
    assert p.baseline.f0_stat.mean == pytest.approx(180, rel=0.05)
    cats = [f.category for f in p.flags]
    assert "voice_energy_change" in cats
    assert all(f.requires_human_review for f in p.flags)
