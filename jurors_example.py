"""Example `jurors_fn` for the CSV runner — minimal stub.

Copy this file, point it at your real juror models, then run:

    python -m voting_kit.csv_runner \\
        --input  data/cleaned/ID.csv \\
        --output data/results/ID_verdicts.csv \\
        --jurors-module voting_kit.jurors_example \\
        --mode auto \\
        --model-path Qwen/Qwen3Guard-Gen-8B \\
        --dtype bfloat16 \\
        --limit 50

The runner imports this module and calls `jurors_fn(row, *, content,
country, language)` once per CSV row. You return three JurorVerdicts
(A: local specialist, B: open-source generalist, C: cloud premium).

`setup()` (optional) is called once before processing — use it to
load models, open API clients, etc.
"""

from voting_kit import JurorVerdict, ViolationCategory


# ── optional one-time init ─────────────────────────────────────────

def setup() -> None:
    """Called once before the CSV is processed. Optional."""
    # Example:
    #   from transformers import pipeline
    #   global _LOCAL_PIPE
    #   _LOCAL_PIPE = pipeline("text-classification",
    #                          model="cardiffnlp/twitter-roberta-base-offensive")
    pass


# ── required: jurors_fn ────────────────────────────────────────────

def jurors_fn(row, *, content, country, language) -> list[JurorVerdict]:
    """Return three JurorVerdicts (A, B, C) for one CSV row.

    `row`     : full row dict from csv.DictReader
    `content` : already-concatenated text from `content_cols`
    `country` : per-row value from `country_col` (or default)
    `language`: per-row value from `language_col` (or default)

    This stub returns dummy verdicts so the pipeline runs end-to-end.
    Replace each block with your real model calls.
    """
    cid = row.get("url") or row.get("id") or content[:32]

    # ── Juror A: local specialist (e.g. fine-tuned BERT for the language) ──
    a = JurorVerdict(
        content_id=cid, juror="A",
        model_name="stub-local-specialist",
        violation=False,
        category=ViolationCategory.none,
        confidence=0.5,
        reasoning="Replace with real local-model output (e.g. classify_direct).",
        language=language,
    )

    # ── Juror B: open-source generalist (e.g. Llama via Together AI) ──
    b = JurorVerdict(
        content_id=cid, juror="B",
        model_name="stub-generalist",
        violation=False,
        category=ViolationCategory.none,
        confidence=0.5,
        reasoning="Replace with real open-source model API call.",
        language=language,
    )

    # ── Juror C: cloud premium (e.g. Claude Haiku) ──
    c = JurorVerdict(
        content_id=cid, juror="C",
        model_name="stub-cloud",
        violation=False,
        category=ViolationCategory.none,
        confidence=0.5,
        reasoning="Replace with real cloud-API call.",
        language=language,
    )

    return [a, b, c]
