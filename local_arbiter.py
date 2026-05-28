"""Local LLM arbiter — load an HF model from disk and use it as the judge.

Designed for guard/safety models that you've already downloaded locally,
e.g. Qwen/Qwen3Guard-Gen-8B or google/shieldgemma-2b. Works with any
instruction-tuned causal LM whose tokenizer ships a chat template.

Same prompt and output contract as `llm_arbiter.call_arbiter`, so the
two are drop-in interchangeable.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import JurorVerdict, FinalVerdict, ViolationCategory
from .prompts import get_prompt, JudgePrompt
from .llm_arbiter import _agreement_str, _fallback_verdict

# Cache loaded models so repeated calls don't reload from disk.
# Key: (model_path, dtype, device) -> (tokenizer, model)
_MODEL_CACHE: dict[tuple, tuple] = {}


def load_local_model(
    model_path: str,
    *,
    dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = True,
):
    """Load a local causal LM + tokenizer (cached across calls).

    Args:
        model_path: Path to the model directory, OR a HuggingFace repo id.
            transformers resolves both. Examples:
                "/home/me/models/Qwen3Guard-Gen-8B"
                "Qwen/Qwen3Guard-Gen-8B"
                "google/shieldgemma-2b"
        dtype: "auto", "bfloat16", "float16", or "float32".
        device_map: passed to from_pretrained — usually "auto" for GPU.
        trust_remote_code: needed for some guard models with custom code.

    Returns: (tokenizer, model)
    """
    cache_key = (str(model_path), dtype, device_map)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    torch_dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, "auto")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    _MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def call_local_arbiter(
    content_id: str,
    content: str,
    verdicts: list[JurorVerdict],
    model_path: str,
    *,
    source: str = "",
    country: str = "",
    language: str = "",
    dtype: str = "auto",
    device_map: str = "auto",
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    trust_remote_code: bool = True,
    prompt: Optional[JudgePrompt] = None,
    system_prompt: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
) -> FinalVerdict:
    """Run a locally-hosted LLM as the arbiter — no API call, no network.

    Args:
        content_id: Unique content identifier.
        content: Original content text (truncated to 3000 chars).
        verdicts: Exactly 3 JurorVerdicts in order [A, B, C].
        model_path: Local path or HF repo id (e.g. "Qwen/Qwen3Guard-Gen-8B",
            "google/shieldgemma-2b", "/data/models/qwen3guard").
        source / country / language: shown to the judge in the prompt.
            `country` is also used to look up the country-specific prompt
            from `voting_kit.prompts` (e.g. "ID" → Indonesia prompt).
        dtype / device_map: forwarded to `load_local_model`.
        max_new_tokens / temperature: generation knobs.
        prompt: explicit JudgePrompt to use. Takes priority over country
            lookup. Use `voting_kit.prompts.register_prompt` to add new
            countries to the registry instead of passing this every call.
        system_prompt / user_prompt_template: legacy fine-grained overrides
            (still honored if `prompt` is None).

    Returns: FinalVerdict. Failures (load/generate/parse) fall back to
        a `requires_human_review=True` verdict — no exceptions raised.
    """
    if len(verdicts) != 3:
        return _fallback_verdict(content_id, verdicts, "Expected 3 verdicts")

    if prompt is not None:
        sys_p = prompt.system
        user_tpl = prompt.user_template
    else:
        country_prompt = get_prompt(country)
        sys_p = system_prompt or country_prompt.system
        user_tpl = user_prompt_template or country_prompt.user_template

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
        tokenizer, model = load_local_model(
            model_path, dtype=dtype, device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        return _fallback_verdict(content_id, verdicts, f"Failed to load local model {model_path}: {e}")

    try:
        raw_response = _generate(
            tokenizer, model, sys_p, user_prompt,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
    except Exception as e:
        return _fallback_verdict(content_id, verdicts, f"Local arbiter generation error: {e}")

    _ = (time.monotonic() - t0) * 1000  # latency, currently unused

    text = raw_response.strip()
    # Some guard models wrap output in ```json fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()

    # Look for the first {...} block if there's surrounding chatter
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

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
            judge_model=f"local:{Path(model_path).name}",
            judged_at=datetime.now(timezone.utc),
            requires_human_review=bool(data.get("requires_human_review", False)),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return _fallback_verdict(
            content_id, verdicts,
            f"Failed to parse local arbiter response: {e}. Raw: {raw_response[:300]}",
        )


def _generate(
    tokenizer,
    model,
    system_prompt: str,
    user_prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate text using the model's chat template."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Some tokenizers (e.g. Gemma) don't accept a system role — fall back
    # to merging system into the first user turn.
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        merged = f"{system_prompt}\n\n{user_prompt}"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": merged}],
            tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def clear_cache() -> None:
    """Drop all cached models to free GPU memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
