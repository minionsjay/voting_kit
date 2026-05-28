"""Unified entry point — pick algorithm-only voting or LLM judge.

`decide()` is the single function most users want. The judge prompt is
picked automatically by `country` (see `voting_kit.prompts`); pass an
explicit `prompt=...` to override.

    from voting_kit import decide

    # Pure algorithm — no model, no API:
    final = decide(verdicts, mode="vote")

    # Local guard model + country-specific prompt:
    final = decide(
        verdicts, content="...",
        mode="local",
        model_path="Qwen/Qwen3Guard-Gen-8B",
        country="ID",                  # picks Indonesia prompt
    )

    # Hybrid: vote first, only call the model on disagreement:
    final = decide(
        verdicts, content="...",
        mode="auto",
        model_path="google/shieldgemma-2b",
        country="TH",
    )
"""

from typing import Literal, Optional

from .schema import JurorVerdict, FinalVerdict
from .voting import majority_vote, weighted_vote
from .llm_arbiter import call_arbiter
from .local_arbiter import call_local_arbiter
from .prompts import JudgePrompt

Mode = Literal["vote", "weighted", "local", "api", "auto"]


def decide(
    verdicts: list[JurorVerdict],
    *,
    mode: Mode = "auto",
    content: str = "",
    content_id: str = "",
    source: str = "",
    country: str = "",
    language: str = "",
    # judge prompt (overrides country lookup)
    prompt: Optional[JudgePrompt] = None,
    # local-judge knobs
    model_path: str = "",
    dtype: str = "auto",
    device_map: str = "auto",
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    trust_remote_code: bool = True,
    # api-judge knobs
    provider: str = "anthropic",
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    no_proxy: bool = False,
    timeout: float = 60.0,
    # auto-mode escalation threshold
    escalate_below_confidence: float = 0.7,
) -> FinalVerdict:
    """One-shot decision over three juror verdicts.

    Modes:
        "vote"     — majority_vote only (weighted as fallback). Algorithm-only.
        "weighted" — weighted_vote only.
        "local"    — call_local_arbiter. Requires `content` and `model_path`.
        "api"      — call_arbiter (cloud). Requires `content` + credentials.
        "auto"     — vote first; escalate to local judge (if model_path given)
                     or api judge when the vote is None or low-confidence.

    Prompt selection (for "local"/"api"/"auto"):
        - explicit `prompt=` wins, else
        - the prompt registered for `country` (see voting_kit.prompts), else
        - the "default" prompt.
    """
    cid = content_id or (verdicts[0].content_id if verdicts else "")

    if mode == "vote":
        return majority_vote(verdicts) or weighted_vote(verdicts) or _no_consensus(verdicts, cid)

    if mode == "weighted":
        return weighted_vote(verdicts) or _no_consensus(verdicts, cid)

    if mode == "local":
        if not model_path:
            raise ValueError('mode="local" requires model_path (e.g. "Qwen/Qwen3Guard-Gen-8B").')
        return call_local_arbiter(
            content_id=cid, content=content, verdicts=verdicts,
            model_path=model_path, source=source, country=country, language=language,
            dtype=dtype, device_map=device_map, max_new_tokens=max_new_tokens,
            temperature=temperature, trust_remote_code=trust_remote_code,
            prompt=prompt,
        )

    if mode == "api":
        return call_arbiter(
            content_id=cid, content=content, verdicts=verdicts,
            source=source, country=country, language=language,
            provider=provider, model=model, base_url=base_url, api_key=api_key,
            no_proxy=no_proxy, timeout=timeout,
            prompt=prompt,
        )

    if mode == "auto":
        voted = majority_vote(verdicts)
        if voted is not None and voted.confidence >= escalate_below_confidence:
            return voted
        # Escalate to LLM judge
        if model_path:
            return call_local_arbiter(
                content_id=cid, content=content, verdicts=verdicts,
                model_path=model_path, source=source, country=country, language=language,
                dtype=dtype, device_map=device_map, max_new_tokens=max_new_tokens,
                temperature=temperature, trust_remote_code=trust_remote_code,
                prompt=prompt,
            )
        return call_arbiter(
            content_id=cid, content=content, verdicts=verdicts,
            source=source, country=country, language=language,
            provider=provider, model=model, base_url=base_url, api_key=api_key,
            no_proxy=no_proxy, timeout=timeout,
            prompt=prompt,
        )

    raise ValueError(f"Unknown mode: {mode!r}")


def _no_consensus(verdicts: list[JurorVerdict], content_id: str) -> FinalVerdict:
    from .schema import ViolationCategory
    parts = []
    for v in verdicts:
        val = "null" if v.violation is None else ("violation" if v.violation else "clean")
        parts.append(f"{v.juror}:{val}")
    return FinalVerdict(
        content_id=content_id,
        final_verdict=False,
        category=ViolationCategory.none,
        confidence=0.0,
        adopted_juror="none",
        adopted_reason="No consensus from voting; no judge configured.",
        juror_agreement=" / ".join(parts),
        reasoning="Voting could not reach a decision and no LLM judge was configured. Escalate to human review.",
        judge_model="voting",
        requires_human_review=True,
    )
