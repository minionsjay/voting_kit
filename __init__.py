"""voting_kit — three-juror voting + local/cloud LLM arbiter with country-aware prompts.

Public API:
    JurorVerdict, FinalVerdict, ViolationCategory   # data models
    majority_vote, weighted_vote                    # algorithm-only
    call_local_arbiter, load_local_model            # local LLM judge (recommended)
    call_arbiter                                    # cloud LLM judge
    decide                                          # one-shot unified entry
    JudgePrompt, get_prompt, register_prompt, list_countries   # country prompts
    BASE_SYSTEM, BASE_USER_TEMPLATE                 # base prompt strings

Quick start (local model + country prompt):

    from voting_kit import decide

    final = decide(
        verdicts,
        content="...原帖文本...",
        mode="local",
        model_path="Qwen/Qwen3Guard-Gen-8B",   # local path or HF id
        country="ID",                          # picks Indonesia prompt
    )
"""

from .schema import JurorVerdict, FinalVerdict, ViolationCategory
from .voting import majority_vote, weighted_vote
from .llm_arbiter import call_arbiter
from .local_arbiter import call_local_arbiter, load_local_model, clear_cache
from .decide import decide
from .csv_runner import run_csv
from .prompts import (
    JudgePrompt,
    get_prompt,
    register_prompt,
    list_countries,
    BASE_SYSTEM,
    BASE_USER_TEMPLATE,
)

__all__ = [
    "JurorVerdict",
    "FinalVerdict",
    "ViolationCategory",
    "majority_vote",
    "weighted_vote",
    "call_arbiter",
    "call_local_arbiter",
    "load_local_model",
    "clear_cache",
    "decide",
    "run_csv",
    "JudgePrompt",
    "get_prompt",
    "register_prompt",
    "list_countries",
    "BASE_SYSTEM",
    "BASE_USER_TEMPLATE",
]
