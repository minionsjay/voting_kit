"""Stage 2 arbiter: LLM judge reviews juror reasoning and makes final verdict.

Self-contained version of the project's `arbiter/llm_arbiter.py` —
reads credentials from environment variables only, no project config
import. Drop this folder into another project as-is.

Prompts come from `voting_kit.prompts` and are looked up by country.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

from .schema import JurorVerdict, FinalVerdict, ViolationCategory
from .prompts import get_prompt, JudgePrompt, BASE_SYSTEM, BASE_USER_TEMPLATE

# Back-compat aliases for callers that imported the constants directly.
ARBITER_SYSTEM_PROMPT = BASE_SYSTEM
ARBITER_USER_PROMPT = BASE_USER_TEMPLATE


def call_arbiter(
    content_id: str,
    content: str,
    verdicts: list[JurorVerdict],
    source: str = "",
    country: str = "",
    language: str = "",
    provider: str = "anthropic",
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    no_proxy: bool = False,
    timeout: float = 60.0,
    prompt: Optional[JudgePrompt] = None,
) -> FinalVerdict:
    """Call the LLM judge to arbitrate between juror verdicts.

    Args:
        content_id: Unique content identifier.
        content: The original content text (truncated to 3000 chars).
        verdicts: Exactly three JurorVerdict objects, in order [A, B, C].
        source / country / language: shown to the judge in the prompt.
            `country` is also used to pick the country-specific prompt
            from `voting_kit.prompts` (e.g. "ID" → Indonesia prompt).
        provider: 'anthropic', 'openai', 'gemini', or 'custom'
            ('custom' speaks the OpenAI chat-completions schema —
             works with most aggregator gateways and local servers).
        model: override the default model id for the chosen provider.
        base_url: custom API endpoint (required for `custom`).
        api_key: override the API key. Falls back to env vars:
            ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY /
            JUROR_C_API_KEY (for `custom`).
        no_proxy: bypass system HTTP_PROXY when calling the API.
        timeout: max seconds per API call.
        prompt: explicit JudgePrompt (overrides the country lookup).

    Returns:
        FinalVerdict. On any failure, falls back to a verdict with
        `requires_human_review=True` and the error in `adopted_reason`.
    """
    if len(verdicts) != 3:
        return _fallback_verdict(content_id, verdicts, "Expected 3 verdicts")

    judge_prompt = prompt or get_prompt(country)
    sys_p = judge_prompt.system
    user_tpl = judge_prompt.user_template

    def _ruling(v: JurorVerdict) -> str:
        if v.violation is True:
            return "VIOLATION"
        if v.violation is False:
            return "CLEAN"
        return "UNCERTAIN"

    user_prompt = user_tpl.format(
        source=source or "unknown",
        country=country or "unknown",
        language=language or "unknown",
        content=content[:3000],
        model_a=verdicts[0].model_name, ruling_a=_ruling(verdicts[0]),
        category_a=verdicts[0].category, confidence_a=verdicts[0].confidence,
        reasoning_a=verdicts[0].reasoning[:1000],
        model_b=verdicts[1].model_name, ruling_b=_ruling(verdicts[1]),
        category_b=verdicts[1].category, confidence_b=verdicts[1].confidence,
        reasoning_b=verdicts[1].reasoning[:1000],
        model_c=verdicts[2].model_name, ruling_c=_ruling(verdicts[2]),
        category_c=verdicts[2].category, confidence_c=verdicts[2].confidence,
        reasoning_c=verdicts[2].reasoning[:1000],
    )

    t0 = time.monotonic()
    try:
        raw_response = _call_llm(
            sys_p, user_prompt, provider, model,
            base_url=base_url, api_key=api_key, no_proxy=no_proxy, timeout=timeout,
        )
    except Exception as e:
        return _fallback_verdict(content_id, verdicts, f"Arbiter API error: {e}")

    _ = (time.monotonic() - t0) * 1000  # latency, currently unused

    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]

    try:
        data = json.loads(text)
        return FinalVerdict(
            content_id=content_id,
            final_verdict=bool(data["final_verdict"]),
            category=ViolationCategory(data.get("category", "none")),
            confidence=float(data.get("confidence", 0.5)),
            adopted_juror=str(data.get("adopted_juror", "none")),
            adopted_reason=str(data.get("adopted_reason", "")),
            juror_agreement=_agreement_str(verdicts),
            reasoning=str(data.get("reasoning", "")),
            judge_model=f"{provider}:{model or 'default'}",
            judged_at=datetime.now(timezone.utc),
            requires_human_review=bool(data.get("requires_human_review", False)),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return _fallback_verdict(content_id, verdicts, f"Failed to parse arbiter response: {e}")


def _call_llm(
    system: str,
    user: str,
    provider: str,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    no_proxy: bool = False,
    timeout: float = 60.0,
) -> str:
    if provider == "custom":
        url = base_url or os.getenv("JUROR_C_BASE_URL", "")
        if not url:
            raise ValueError("Custom arbiter requires base_url. Set JUROR_C_BASE_URL env var.")
        if not url.endswith("/chat/completions"):
            url = url.rstrip("/") + "/chat/completions"
        key = api_key or os.getenv("JUROR_C_API_KEY", "")
        model = model or "gpt-4o"
        kwargs = {"timeout": timeout, "verify": False}
        if no_proxy:
            kwargs["trust_env"] = False
        with httpx.Client(**kwargs) as client:
            resp = client.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    elif provider == "anthropic":
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        model = model or "claude-sonnet-4-6-20250514"
        url = base_url or "https://api.anthropic.com/v1/messages"
        kwargs = {"timeout": timeout}
        if no_proxy:
            kwargs["trust_env"] = False
        with httpx.Client(**kwargs) as client:
            resp = client.post(
                url,
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                json={
                    "model": model,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "temperature": 0.2,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]

    elif provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        model = model or "gpt-4o"
        url = base_url or "https://api.openai.com/v1/chat/completions"
        kwargs = {"timeout": timeout}
        if no_proxy:
            kwargs["trust_env"] = False
        with httpx.Client(**kwargs) as client:
            resp = client.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    elif provider == "gemini":
        key = api_key or os.getenv("GOOGLE_API_KEY", "")
        model = model or "gemini-2.5-flash"
        kwargs = {"timeout": timeout}
        if no_proxy:
            kwargs["trust_env"] = False
        with httpx.Client(**kwargs) as client:
            resp = client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                json={
                    "contents": [{"parts": [{"text": f"{system}\n\n{user}"}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
                },
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    else:
        raise ValueError(f"Unknown arbiter provider: {provider}")


def _fallback_verdict(content_id: str, verdicts: list[JurorVerdict], reason: str) -> FinalVerdict:
    return FinalVerdict(
        content_id=content_id,
        final_verdict=False,
        category=ViolationCategory.none,
        confidence=0.0,
        adopted_juror="none",
        adopted_reason=reason,
        juror_agreement=_agreement_str(verdicts) if verdicts else "",
        reasoning=f"Arbiter unavailable: {reason}. Requires human review.",
        judge_model="fallback",
        judged_at=datetime.now(timezone.utc),
        requires_human_review=True,
    )


def _agreement_str(verdicts: list[JurorVerdict]) -> str:
    parts = []
    for v in verdicts:
        val = "null" if v.violation is None else ("violation" if v.violation else "clean")
        parts.append(f"{v.juror}:{val}")
    return " / ".join(parts)
