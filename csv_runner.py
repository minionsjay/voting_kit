"""CSV input/output for voting_kit.

Read a CSV of posts, run three jurors per row, vote/judge, write a
results CSV. Designed to work with the project's cleaned CSVs
(data/cleaned/{COUNTRY}.csv) but is generic — any CSV with text in
some column will do.

Usage:
    from voting_kit.csv_runner import run_csv

    def my_jurors(row, *, content, country, language):
        # Call your three juror models here, return three JurorVerdicts.
        return [verdict_a, verdict_b, verdict_c]

    stats = run_csv(
        input_csv="data/cleaned/ID.csv",
        output_csv="data/results/ID_verdicts.csv",
        jurors_fn=my_jurors,
        mode="auto",
        model_path="Qwen/Qwen3Guard-Gen-8B",
        # column mapping (defaults match data/cleaned/*.csv):
        content_cols=["title", "body"],
        country_col="country",
    )
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Optional

from .schema import JurorVerdict, FinalVerdict
from .decide import decide
from .prompts import JudgePrompt

JurorsFn = Callable[..., list[JurorVerdict]]

DEFAULT_OUTPUT_COLUMNS = [
    "content_id", "country", "language",
    "final_verdict", "category", "confidence",
    "adopted_juror", "juror_agreement",
    "judge_model", "requires_human_review",
    "reasoning", "adopted_reason",
    "juror_a_violation", "juror_a_confidence", "juror_a_reasoning",
    "juror_b_violation", "juror_b_confidence", "juror_b_reasoning",
    "juror_c_violation", "juror_c_confidence", "juror_c_reasoning",
]


def run_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    jurors_fn: JurorsFn,
    *,
    # decide() knobs
    mode: str = "auto",
    model_path: str = "",
    country: str = "",
    language: str = "",
    prompt: Optional[JudgePrompt] = None,
    dtype: str = "auto",
    device_map: str = "auto",
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    trust_remote_code: bool = True,
    provider: str = "anthropic",
    api_model: str = "",
    api_key: str = "",
    base_url: str = "",
    # csv knobs
    content_cols: list[str] | None = None,
    country_col: str = "country",
    language_col: str = "language",
    id_col: str = "",
    limit: int = 0,
    skip: int = 0,
    resume: bool = True,
    extra_input_cols: list[str] | None = None,
    pass_row_to_jurors: bool = True,
) -> dict:
    """Run the jury+judge pipeline over a CSV file.

    Args:
        input_csv: Path to input CSV. UTF-8 (with optional BOM) is fine.
        output_csv: Path where results are written. Created/appended.
        jurors_fn: Callback returning three JurorVerdicts for one row.
            Signature: `jurors_fn(row, *, content, country, language) -> [v_a, v_b, v_c]`
            (or just `jurors_fn(content, country, language)` if you set
            `pass_row_to_jurors=False`.)
        mode/model_path/country/language/prompt/...: forwarded to `decide()`.
            `country` here is a default — the per-row value from `country_col`
            wins when present.
        content_cols: input columns to concatenate into the content text.
            Default ["title", "body"] (matches the project's cleaned CSVs).
        country_col / language_col: where to read per-row country/language.
            Pass empty string ("") to disable lookup.
        id_col: column to use as content_id. Empty → use row index.
        limit: process at most this many rows (0 = all).
        skip: skip the first N rows.
        resume: if output_csv exists, skip rows whose content_id is already
            present (by `content_id` column).
        extra_input_cols: copy these additional columns through into the
            output CSV unchanged.
        pass_row_to_jurors: if True (default), `jurors_fn` is called with
            the full row dict + kwargs; if False, only positional args
            `(content, country, language)`.

    Returns:
        dict with keys:
            "processed", "violations", "clean", "human_review",
            "skipped_resume", "errors", "output_csv".
    """
    in_path = Path(input_csv)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if content_cols is None:
        content_cols = ["title", "body"]
    if extra_input_cols is None:
        extra_input_cols = []

    output_columns = DEFAULT_OUTPUT_COLUMNS + [
        c for c in extra_input_cols if c not in DEFAULT_OUTPUT_COLUMNS
    ]

    seen_ids: set[str] = set()
    if resume and out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("content_id", "")
                if cid:
                    seen_ids.add(cid)

    write_header = not (out_path.exists() and out_path.stat().st_size > 0)

    stats = {
        "processed": 0, "violations": 0, "clean": 0,
        "human_review": 0, "skipped_resume": 0, "errors": 0,
    }

    # utf-8-sig handles the BOM that Reddit/Excel exports often carry.
    with in_path.open("r", encoding="utf-8-sig", newline="") as fin, \
         out_path.open("a", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=output_columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for idx, row in enumerate(reader):
            if idx < skip:
                continue
            if limit and stats["processed"] >= limit:
                break

            content_id = (row.get(id_col, "") if id_col else "") or f"{in_path.stem}:{idx}"
            if content_id in seen_ids:
                stats["skipped_resume"] += 1
                continue

            content = _build_content(row, content_cols)
            if not content.strip():
                stats["errors"] += 1
                continue

            row_country = row.get(country_col, "").strip() if country_col else ""
            row_lang = row.get(language_col, "").strip() if language_col else ""
            effective_country = row_country or country
            effective_language = row_lang or language

            try:
                if pass_row_to_jurors:
                    verdicts = jurors_fn(
                        row,
                        content=content,
                        country=effective_country,
                        language=effective_language,
                    )
                else:
                    verdicts = jurors_fn(content, effective_country, effective_language)
            except Exception as e:
                stats["errors"] += 1
                _log(f"[{idx}] jurors_fn error: {e}")
                continue

            # Pad / pin content_id so decide() and downstream are consistent.
            verdicts = _normalise_verdicts(verdicts, content_id, effective_language)

            try:
                final = decide(
                    verdicts,
                    mode=mode,
                    content=content,
                    content_id=content_id,
                    source=row.get("source", ""),
                    country=effective_country,
                    language=effective_language,
                    prompt=prompt,
                    model_path=model_path,
                    dtype=dtype,
                    device_map=device_map,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    trust_remote_code=trust_remote_code,
                    provider=provider,
                    model=api_model,
                    api_key=api_key,
                    base_url=base_url,
                )
            except Exception as e:
                stats["errors"] += 1
                _log(f"[{idx}] decide() error: {e}")
                continue

            out_row = _final_to_row(final, verdicts, effective_country, effective_language)
            for c in extra_input_cols:
                out_row[c] = row.get(c, "")
            writer.writerow(out_row)
            fout.flush()

            stats["processed"] += 1
            if final.requires_human_review:
                stats["human_review"] += 1
            elif final.final_verdict:
                stats["violations"] += 1
            else:
                stats["clean"] += 1

            if stats["processed"] % 50 == 0:
                _log(f"  …{stats['processed']} rows processed "
                     f"(violations={stats['violations']}, clean={stats['clean']}, "
                     f"review={stats['human_review']})")

    stats["output_csv"] = str(out_path)
    _log(f"Done. {stats}")
    return stats


# ── helpers ────────────────────────────────────────────────────────

def _build_content(row: dict, cols: list[str]) -> str:
    parts = []
    for c in cols:
        v = (row.get(c) or "").strip()
        if v:
            parts.append(v)
    return "\n\n".join(parts)


def _normalise_verdicts(
    verdicts: list[JurorVerdict],
    content_id: str,
    language: str,
) -> list[JurorVerdict]:
    """Ensure exactly 3 verdicts, in [A, B, C] order, with content_id set.

    Missing slots are filled with `violation=None` placeholders so the
    downstream code never crashes on a malformed jurors_fn return.
    """
    by_juror = {v.juror: v for v in verdicts if v.juror in ("A", "B", "C")}
    out = []
    for j in ("A", "B", "C"):
        v = by_juror.get(j)
        if v is None:
            v = JurorVerdict(
                content_id=content_id, juror=j, model_name="missing",
                violation=None, language=language,
                reasoning="No verdict returned by jurors_fn.",
            )
        else:
            v.content_id = content_id
            if not v.language:
                v.language = language
        out.append(v)
    return out


def _final_to_row(
    final: FinalVerdict,
    verdicts: list[JurorVerdict],
    country: str,
    language: str,
) -> dict:
    d = asdict(final)
    d["country"] = country
    d["language"] = language
    d["category"] = (
        final.category.value
        if hasattr(final.category, "value") else str(final.category)
    )
    # flatten dates
    if "judged_at" in d:
        d.pop("judged_at", None)

    for v in verdicts:
        suffix = v.juror.lower()
        d[f"juror_{suffix}_violation"] = (
            "" if v.violation is None else ("true" if v.violation else "false")
        )
        d[f"juror_{suffix}_confidence"] = f"{v.confidence:.3f}"
        d[f"juror_{suffix}_reasoning"] = (v.reasoning or "")[:1000]
    return d


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── CLI ────────────────────────────────────────────────────────────
#
# Run from repo root. Provide a Python file that defines `jurors_fn`:
#
#     python -m voting_kit.csv_runner \
#         --input data/cleaned/ID.csv \
#         --output data/results/ID_verdicts.csv \
#         --jurors-module my_jurors \
#         --mode auto \
#         --model-path Qwen/Qwen3Guard-Gen-8B \
#         --limit 100

def _main() -> None:
    import argparse
    import importlib

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--jurors-module", required=True,
                   help="Python module path that exports `jurors_fn` (and "
                        "optionally `setup`).")
    p.add_argument("--mode", default="auto",
                   choices=["vote", "weighted", "local", "api", "auto"])
    p.add_argument("--model-path", default="")
    p.add_argument("--country", default="",
                   help="Default country if CSV row has no country column")
    p.add_argument("--language", default="")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--content-cols", default="title,body")
    p.add_argument("--country-col", default="country")
    p.add_argument("--language-col", default="language")
    p.add_argument("--id-col", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    mod = importlib.import_module(args.jurors_module)
    if hasattr(mod, "setup"):
        mod.setup()
    jurors_fn = getattr(mod, "jurors_fn")

    stats = run_csv(
        input_csv=args.input,
        output_csv=args.output,
        jurors_fn=jurors_fn,
        mode=args.mode,
        model_path=args.model_path,
        country=args.country,
        language=args.language,
        dtype=args.dtype,
        content_cols=[c.strip() for c in args.content_cols.split(",") if c.strip()],
        country_col=args.country_col,
        language_col=args.language_col,
        id_col=args.id_col,
        limit=args.limit,
        skip=args.skip,
        resume=not args.no_resume,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _main()
