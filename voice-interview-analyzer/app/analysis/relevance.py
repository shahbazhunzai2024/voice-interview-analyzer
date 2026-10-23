"""Answer relevance analysis (Section 5.7).

``LexicalRelevanceScorer`` is a transparent, offline baseline: it compares
content words of the question and answer (with light stemming and a small
synonym/paraphrase-insensitive fallback). It is intentionally conservative
and always returns rationale, evidence and uncertainty.

Replace/augment with embeddings + a structured LLM rubric (Section 6) by
implementing ``RelevanceScorer`` — the pipeline only depends on the protocol.
A low score is a *review cue*, never a judgement that a question was avoided.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

STOPWORDS = set(
    """
    a an the and or but if then so to of in on at by for with about as into
    from up down over under again further is are was were be been being am
    do does did doing have has had having i me my mine we our ours you your
    yours he him his she her it its they them their what which who whom this
    that these those there here when where why how all any both each few more
    most other some such no nor not only own same than too very can will just
    should could would may might must shall tell walk through describe explain
    give example please like really think know yeah um uh okay well also get
    got one thing things lot kind sort mean actually basically
    """.split()
)


def _stem(word: str) -> str:
    for suffix in ("ing", "edly", "ed", "ies", "es", "s", "ly", "ment", "ness"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def content_terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z'+#.-]*", text.lower())
    return {_stem(w.strip(".'")) for w in words if w not in STOPWORDS and len(w) > 2}


@dataclass
class RelevanceResult:
    label: str  # relevant | partially_relevant | off_topic | insufficient_context
    score: float | None
    rationale: str
    evidence: dict = field(default_factory=dict)
    uncertainty: str = ""


class RelevanceScorer(Protocol):
    def score(self, question: str, answer: str) -> RelevanceResult: ...


class LexicalRelevanceScorer:
    def __init__(self, relevant_at: float = 0.34, partial_at: float = 0.12,
                 min_answer_words: int = 6) -> None:
        self.relevant_at = relevant_at
        self.partial_at = partial_at
        self.min_answer_words = min_answer_words

    def score(self, question: str, answer: str) -> RelevanceResult:
        q_terms = content_terms(question)
        a_terms = content_terms(answer)
        excerpt = answer if len(answer) <= 200 else answer[:197] + "..."
        if len(answer.split()) < self.min_answer_words or not q_terms:
            return RelevanceResult(
                "insufficient_context", None,
                "Answer (or question) too short to assess relevance automatically.",
                {"question": question, "answer_excerpt": excerpt},
                "Short or fragmentary turns are common; review manually.",
            )
        shared = sorted(q_terms & a_terms)
        coverage = len(shared) / len(q_terms)
        if coverage >= self.relevant_at:
            label = "relevant"
            why = f"Answer addresses {len(shared)} of {len(q_terms)} key question terms."
        elif coverage >= self.partial_at:
            label = "partially_relevant"
            why = f"Answer overlaps only partly with the question ({len(shared)}/{len(q_terms)} key terms)."
        else:
            label = "off_topic"
            why = "Few or no key question terms appear in the answer."
        return RelevanceResult(
            label,
            round(coverage, 3),
            why,
            {
                "question": question,
                "answer_excerpt": excerpt,
                "shared_terms": shared,
                "question_terms": sorted(q_terms),
            },
            "Lexical baseline: paraphrases, examples, and transcription errors can "
            "lower the score for a relevant answer.",
        )


def get_default_scorer() -> RelevanceScorer:
    return LexicalRelevanceScorer()
