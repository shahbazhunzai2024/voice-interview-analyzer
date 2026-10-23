"""Interviewer question detection from live transcript (Section 5.5).

Rule-based baseline that recognises direct questions ("What is...?"),
indirect prompts ("Tell me about...", "Walk me through..."), and compound
questions (several question sentences in one turn). Each candidate carries a
confidence so a classifier/LLM can later replace or re-rank it.
"""

from __future__ import annotations

import re
import uuid

from app.schemas.events import Question, TranscriptSegment

WH_WORDS = ("what", "why", "how", "when", "where", "which", "who", "whom", "whose")
AUX_STARTS = (
    "do", "does", "did", "is", "are", "was", "were", "can", "could", "would",
    "will", "should", "have", "has", "had", "may", "might", "shall",
)
INDIRECT_PATTERNS = [
    r"\btell (me|us) (about|how|why|what|more)\b",
    r"\bwalk (me|us) through\b",
    r"\b(describe|explain|elaborate on|talk (me |us )?(about|through))\b",
    r"\bgive (me|us) an example\b",
    r"\bi('d| would) (like|love) to (hear|know|understand)\b",
    r"\b(share|take me through) (an|a|your|some)\b",
    r"\bi('m| am) (curious|interested) (about|in|to)\b",
]
# Phrases that look like questions but usually are not prompts to the candidate.
NON_QUESTION_PATTERNS = [
    r"^(i wonder|i don'?t know (what|why|how))\b(?!.*\?)",
    r"^(that'?s|this is) (what|why|how)\b",
    r"^(great|okay|ok|thanks|thank you|sure|right)[.!]?$",
]

_SENTENCE_RE = re.compile(r"[^.?!]+[.?!]?")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


def classify_sentence(sentence: str) -> tuple[bool, float, str]:
    """Return (is_question, confidence, kind) for one sentence."""
    s = sentence.strip()
    lower = s.lower().lstrip("\"'(").strip()
    for pat in NON_QUESTION_PATTERNS:
        if re.search(pat, lower):
            return False, 0.0, "direct"

    ends_q = s.endswith("?")
    first = re.split(r"\W+", lower, maxsplit=1)[0] if lower else ""
    starts_wh = first in WH_WORDS
    starts_aux = first in AUX_STARTS
    indirect = any(re.search(p, lower) for p in INDIRECT_PATTERNS)

    if ends_q and (starts_wh or starts_aux):
        return True, 0.95, "direct"
    if ends_q:
        return True, 0.8, "direct"
    if indirect:
        return True, 0.75, "indirect"
    if starts_wh and len(lower.split()) >= 3:
        # STT output often drops '?'
        return True, 0.6, "direct"
    if starts_aux and first not in {"do", "is", "are"} and len(lower.split()) >= 4:
        return True, 0.5, "direct"
    return False, 0.0, "direct"


def detect_questions(segment: TranscriptSegment, min_confidence: float = 0.5) -> list[Question]:
    """Detect question candidates in one interviewer transcript segment.

    Multiple question sentences in one turn are merged into a single
    ``compound`` question so one candidate answer can address them together.
    """
    if segment.role != "interviewer":
        return []
    hits: list[tuple[str, float, str]] = []
    for sentence in split_sentences(segment.text):
        ok, conf, kind = classify_sentence(sentence)
        if ok and conf >= min_confidence:
            hits.append((sentence, conf, kind))
    if not hits:
        return []

    if len(hits) == 1:
        text, conf, kind = hits[0]
    else:
        text = " ".join(h[0] for h in hits)
        conf = max(h[1] for h in hits)
        kind = "compound"
    return [
        Question(
            question_id=f"q-{uuid.uuid4().hex[:8]}",
            text=text,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            confidence=round(conf, 2),
            kind=kind,
            source_segment_id=segment.segment_id,
        )
    ]


def is_question_text(text: str) -> bool:
    return any(classify_sentence(s)[0] for s in split_sentences(text))
