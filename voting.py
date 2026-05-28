"""Stage 1 arbiter: rule-based voting over three juror verdicts.

Phase 1 of the arbitration strategy — no LLM required, no API key.
Use this before you have enough labeled data to justify an LLM judge,
or as a cheap pre-filter in front of `llm_arbiter.call_arbiter`.
"""

from .schema import JurorVerdict, FinalVerdict, ViolationCategory


def majority_vote(verdicts: list[JurorVerdict]) -> FinalVerdict | None:
    """Simple majority vote over 3 jurors.

    Returns None when no consensus can be reached (1-1-1 split or too
    many `None` rulings) — the caller should escalate those to an LLM
    judge or to human review.
    """
    if len(verdicts) != 3:
        return None

    violations = [v.violation for v in verdicts]

    yes = sum(1 for v in violations if v is True)
    no = sum(1 for v in violations if v is False)
    nulls = sum(1 for v in violations if v is None)

    if nulls >= 2:
        return None

    if yes == 3:
        return _make_verdict(verdicts, True, "consensus", "All three jurors agree: violation.")
    if no == 3:
        return _make_verdict(verdicts, False, "consensus", "All three jurors agree: clean.")

    if yes == 2:
        winner = [v for v in verdicts if v.violation is True]
        return _make_verdict(
            verdicts,
            True,
            winner[0].juror if len(winner) == 1 else "majority",
            f"Majority (2:1) found violation. Adopting {' & '.join(v.juror for v in winner)}.",
        )
    if no == 2:
        winner = [v for v in verdicts if v.violation is False]
        return _make_verdict(
            verdicts,
            False,
            winner[0].juror if len(winner) == 1 else "majority",
            f"Majority (2:1) found no violation. Adopting {' & '.join(v.juror for v in winner)}.",
        )

    return None


def weighted_vote(
    verdicts: list[JurorVerdict],
    weights: dict[str, float] | None = None,
    min_confidence: float = 0.5,
    review_threshold: float = 0.7,
) -> FinalVerdict | None:
    """Weighted vote with per-juror weights.

    Defaults reflect the project's prior on juror quality:
        A (local specialist): 1.5  — highest weight for local-language nuance
        B (open-source generalist): 1.0
        C (cloud premium): 1.2

    Returns None when the net confidence falls below `min_confidence`.
    Sets `requires_human_review=True` when below `review_threshold`.
    """
    if weights is None:
        weights = {"A": 1.5, "B": 1.0, "C": 1.2}

    score = 0.0
    max_score = 0.0
    voting_details: list[str] = []

    for v in verdicts:
        w = weights.get(v.juror, 1.0)
        if v.violation is True:
            score += w * v.confidence
            max_score += w * v.confidence
            voting_details.append(f"{v.juror}:violation({v.confidence:.2f}*{w})")
        elif v.violation is False:
            score -= w * v.confidence
            max_score += w * v.confidence
            voting_details.append(f"{v.juror}:clean({v.confidence:.2f}*{w})")
        else:
            voting_details.append(f"{v.juror}:null")

    if max_score == 0:
        return None

    confidence = min(abs(score) / max_score, 1.0)

    if confidence < min_confidence:
        return None

    final_violation = score > 0

    best = max(verdicts, key=lambda v: v.confidence if v.violation is not None else -1)

    return FinalVerdict(
        content_id=verdicts[0].content_id,
        final_verdict=final_violation,
        category=best.category if best.violation is not None else ViolationCategory.none,
        confidence=confidence,
        adopted_juror=best.juror,
        adopted_reason=f"Weighted vote: {', '.join(voting_details)}. Score={score:.3f}",
        juror_agreement=_agreement_str(verdicts),
        reasoning="Weighted voting (Stage 1).",
        judge_model="voting",
        requires_human_review=confidence < review_threshold,
    )


def _make_verdict(
    verdicts: list[JurorVerdict],
    final_violation: bool,
    adopted: str,
    reason: str,
) -> FinalVerdict:
    majority = [v for v in verdicts if v.violation == final_violation]
    best = max(
        majority or [v for v in verdicts if v.violation is not None],
        key=lambda v: v.confidence,
    )
    return FinalVerdict(
        content_id=verdicts[0].content_id,
        final_verdict=final_violation,
        category=best.category,
        confidence=best.confidence,
        adopted_juror=adopted,
        adopted_reason=reason,
        juror_agreement=_agreement_str(verdicts),
        reasoning="Majority voting (Stage 1).",
        judge_model="voting",
    )


def _agreement_str(verdicts: list[JurorVerdict]) -> str:
    parts = []
    for v in verdicts:
        val = "null" if v.violation is None else ("violation" if v.violation else "clean")
        parts.append(f"{v.juror}:{val}")
    return " / ".join(parts)
