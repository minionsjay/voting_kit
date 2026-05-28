"""End-to-end example: local model judge with country-aware prompts.

Run from the repo root. The default `--mode vote` works without any
ML deps; the rest need `pip install torch transformers accelerate` and
a model on disk (or HF cache).

    # Algorithm-only smoke test (no model needed):
    python -m voting_kit.example

    # Local guard model + Indonesia prompt:
    python -m voting_kit.example \\
        --mode local --country ID \\
        --model-path Qwen/Qwen3Guard-Gen-8B --dtype bfloat16

    # Local model + Thailand prompt, model loaded from disk:
    python -m voting_kit.example \\
        --mode local --country TH \\
        --model-path /data/models/shieldgemma-2b

    # Vote-first; only call the model on disagreement:
    python -m voting_kit.example \\
        --mode auto --country BR \\
        --model-path google/shieldgemma-2b

    # Add a custom country prompt (Malaysia) and use it:
    python -m voting_kit.example --mode show-prompt --country MY-custom
"""

import argparse

from voting_kit import (
    JurorVerdict,
    ViolationCategory,
    JudgePrompt,
    BASE_SYSTEM,
    BASE_USER_TEMPLATE,
    decide,
    get_prompt,
    register_prompt,
    list_countries,
)


# ── Demo verdicts per country ──────────────────────────────────────
#
# In your real pipeline these come from your three jurors. Here we
# hardcode plausible examples per country so you can see how the
# country-specific prompt steers the judge.

DEMOS = {
    "ID": dict(
        content_id="post_id_001",
        content="Dasar bangsat lo, semua orang Cina memang gitu.",
        country="ID",
        language="id",
        source="reddit:r/indonesia",
        verdicts=[
            ("A", "IndoBERT-hate-speech",   True,  "hate_speech", 0.91,
             "Indonesian slur 'bangsat' followed by ethnic generalisation against Chinese-Indonesians."),
            ("B", "Llama-3.1-70B",           False, "none",        0.55,
             "Reads as casual venting between users."),
            ("C", "claude-haiku-4-5",        True,  "hate_speech", 0.83,
             "Slur + ethnic slur 'cina' in collective form is hateful."),
        ],
    ),
    "TH": dict(
        content_id="post_th_001",
        content="กษัตริย์ไทยเป็นแค่หุ่นเชิด ระบอบนี้ต้องล้ม",
        country="TH",
        language="th",
        source="pantip",
        verdicts=[
            ("A", "wangchanberta-safety",    True,  "political",   0.78,
             "Direct attack on the monarchy — falls under Article 112 sensitivity."),
            ("B", "Llama-3.1-70B",           True,  "political",   0.62,
             "Calls for system overthrow; political content."),
            ("C", "claude-haiku-4-5",        True,  "political",   0.80,
             "Insulting monarchy + incitement; sensitive under Thai law."),
        ],
    ),
    "BR": dict(
        content_id="post_br_001",
        content="Esses macacos do nordeste só sabem votar errado.",
        country="BR",
        language="pt",
        source="reddit:r/brasil",
        verdicts=[
            ("A", "TeenyTinyLlama-HateBR",   True,  "hate_speech", 0.93,
             "'macacos' targeting Northeasterners is a known racist trope in BR."),
            ("B", "Llama-3.1-70B",           True,  "hate_speech", 0.71,
             "Regional + racial slur combined with political stereotyping."),
            ("C", "claude-haiku-4-5",        True,  "hate_speech", 0.84,
             "Racial slur with regional discrimination."),
        ],
    ),
    "MX": dict(
        content_id="post_mx_001",
        content="Estos pinches indios nacos no entienden nada.",
        country="MX",
        language="es",
        source="reddit:r/mexico",
        verdicts=[
            ("A", "beto-sentiment",          None,  "none",        0.0,
             "Sentiment model can't classify hate; returned negative sentiment only."),
            ("B", "Llama-3.1-70B",           True,  "hate_speech", 0.74,
             "'indios' + 'nacos' targeting indigenous Mexicans is derogatory."),
            ("C", "claude-haiku-4-5",        True,  "hate_speech", 0.80,
             "Classist + anti-indigenous slurs combined."),
        ],
    ),
}


def to_verdicts(spec: dict) -> list[JurorVerdict]:
    out = []
    for juror, model_name, violation, category, conf, reasoning in spec["verdicts"]:
        out.append(JurorVerdict(
            content_id=spec["content_id"],
            juror=juror,
            model_name=model_name,
            violation=violation,
            category=ViolationCategory(category),
            confidence=conf,
            reasoning=reasoning,
            language=spec["language"],
        ))
    return out


def run_decide(args, spec: dict) -> None:
    verdicts = to_verdicts(spec)
    print(f"\n══ {spec['country']} (mode={args.mode}) ══")
    print(f"Content: {spec['content']}\n")

    final = decide(
        verdicts,
        mode=args.mode,
        content=spec["content"],
        source=spec["source"],
        country=spec["country"],
        language=spec["language"],
        model_path=args.model_path,
        provider=args.provider,
        dtype=args.dtype,
    )

    print(f"final_verdict        : {final.final_verdict}")
    print(f"category             : {final.category}")
    print(f"confidence           : {final.confidence:.3f}")
    print(f"adopted_juror        : {final.adopted_juror}")
    print(f"juror_agreement      : {final.juror_agreement}")
    print(f"judge_model          : {final.judge_model}")
    print(f"requires_human_review: {final.requires_human_review}")
    print(f"reasoning            : {final.reasoning}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="vote",
                        choices=["vote", "weighted", "local", "api", "auto", "show-prompt"])
    parser.add_argument("--model-path", default="",
                        help="Local model path or HF repo id (for local/auto)")
    parser.add_argument("--country", default="",
                        help="Country code: ID/TH/BR/MX/SG/SA/TR/ZA/AE/PH/VN. "
                             "Empty = run all demos.")
    parser.add_argument("--provider", default="anthropic",
                        help="API provider for api/auto fallback")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"])
    args = parser.parse_args()

    # Demo: register a custom country prompt at runtime
    register_prompt("MY-custom", JudgePrompt(
        system=BASE_SYSTEM + (
            "\n## Country focus: Malaysia\n"
            "- Multilingual (Bahasa Melayu, English, Mandarin, Tamil); code-switching is normal.\n"
            "- 3R sensitivity: Race, Religion, Royalty — flag explicit attacks.\n"
            "- 'kafir' / 'bangsa pendatang' in racial framings → hate_speech.\n"
        ),
        user_template=BASE_USER_TEMPLATE,
    ))

    if args.mode == "show-prompt":
        target = args.country or "default"
        p = get_prompt(target)
        print(f"── prompt for country={target!r} ──\n")
        print(p.system)
        print("\nRegistered countries:", list_countries())
        return

    if args.country:
        if args.country not in DEMOS:
            print(f"No demo content for {args.country}; "
                  f"available: {sorted(DEMOS.keys())}")
            return
        run_decide(args, DEMOS[args.country])
    else:
        for spec in DEMOS.values():
            run_decide(args, spec)


if __name__ == "__main__":
    main()
