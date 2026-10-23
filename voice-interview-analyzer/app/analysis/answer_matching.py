"""Associate candidate answers with interviewer questions (Section 5.5).

Strategy (stored in ``association_method``):

* ``most_recent_preceding`` — the latest question that started before the
  answer and is within ``max_gap_ms``.
* ``continuation`` — consecutive candidate segments after the same question
  are treated as one answer turn (handled by the pipeline aggregating turns).
* Clarifying questions from the candidate are tagged ``clarification`` and
  do not consume the question.
* If nothing qualifies, the answer is explicitly ``unresolved`` rather than
  forced onto a wrong question.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.analysis.question_detection import is_question_text
from app.schemas.events import Question, TranscriptSegment


@dataclass
class MatchResult:
    question: Question | None
    status: str  # paired | unresolved | clarification
    method: str


class AnswerMatcher:
    def __init__(self, max_gap_ms: int = 120_000) -> None:
        self.max_gap_ms = max_gap_ms

    def match(self, answer: TranscriptSegment, questions: list[Question]) -> MatchResult:
        if answer.role != "candidate":
            raise ValueError("only candidate segments can be matched as answers")

        preceding = [q for q in questions if q.start_ms <= answer.start_ms]
        if not preceding:
            return MatchResult(None, "unresolved", "no_preceding_question")
        q = max(preceding, key=lambda x: x.start_ms)

        if is_question_text(answer.text) and len(answer.text.split()) <= 25:
            return MatchResult(q, "clarification", "candidate_clarifying_question")

        if answer.start_ms - q.end_ms > self.max_gap_ms:
            return MatchResult(None, "unresolved", "gap_exceeds_limit")

        return MatchResult(q, "paired", "most_recent_preceding")
