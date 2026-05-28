"""Self-contained data models used by the voting kit.

Mirrors the project's pipeline.schema but kept here so the kit can be
dropped into other projects without dragging the rest of the codebase.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class ViolationCategory(str, Enum):
    hate_speech = "hate_speech"
    violence = "violence"
    adult = "adult"
    fraud = "fraud"
    illegal = "illegal"
    political = "political"
    none = "none"


@dataclass
class JurorVerdict:
    """A single juror's judgment."""

    content_id: str
    juror: str  # "A", "B", "C"
    model_name: str
    violation: Optional[bool]  # None = unable to judge
    category: ViolationCategory = ViolationCategory.none
    confidence: float = 0.0
    reasoning: str = ""
    language: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    judged_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FinalVerdict:
    """Arbiter's final judgment after reviewing all jurors."""

    content_id: str
    final_verdict: bool
    category: ViolationCategory
    confidence: float
    adopted_juror: str  # "A", "B", "C", "consensus", "majority", "none"
    adopted_reason: str
    juror_agreement: str  # e.g. "A:violation / B:clean / C:violation"
    reasoning: str
    judge_model: str
    judged_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    requires_human_review: bool = False
