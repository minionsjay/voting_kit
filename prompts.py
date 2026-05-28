"""Country-aware prompts for the LLM judge.

Why this exists:
    Different countries / languages have different sensitivities — what's
    a normal political opinion in one place is a violation in another, and
    local slang/dog-whistles only one language model would notice.

How to use:
    from voting_kit.prompts import get_prompt, register_prompt, JudgePrompt

    # Use a built-in country prompt
    p = get_prompt("ID")        # Indonesia
    p = get_prompt("default")   # generic fallback

    # Override or add a new country
    register_prompt("MY", JudgePrompt(
        system=BASE_SYSTEM + "\\n## Country focus: Malaysia ...",
        user_template=BASE_USER_TEMPLATE,
    ))

The judge call (`decide(..., mode="local", country="ID")`) automatically
looks up the right prompt for that country and falls back to "default"
if none is registered.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class JudgePrompt:
    """A pair of (system_prompt, user_prompt_template) used by the judge.

    The user_template MUST contain these placeholders (used by .format()):
        {source} {country} {language} {content}
        {model_a} {ruling_a} {category_a} {confidence_a} {reasoning_a}
        {model_b} {ruling_b} {category_b} {confidence_b} {reasoning_b}
        {model_c} {ruling_c} {category_c} {confidence_c} {reasoning_c}
    """
    system: str
    user_template: str


# ── Base prompt: shared scaffold for every country ──────────────────

BASE_SYSTEM = """You are the Chief Justice of a content moderation panel. Three jurors (A, B, C) have each reviewed a piece of internet content and rendered their verdicts with reasoning.

Your job is to:
1. Read the original content carefully
2. Review each juror's reasoning
3. Make the FINAL verdict — violation or not
4. Explain which juror you agree with and why

## Violation Categories
- hate_speech: targeting race, ethnicity, religion, gender, sexual orientation, disability
- violence:   threats, terrorism, incitement to physical harm
- adult:      pornography, sexual solicitation
- fraud:      scams, phishing, financial fraud, impersonation
- illegal:    drugs, weapons, stolen data, criminal coordination
- political:  incitement to overthrow government, extremist content (judge by country context)
- none:       not a violation

## Rules
- You are NOT bound by majority vote. If 2 jurors say "clean" but their reasoning is weak while the 1 dissenter has strong, specific evidence — adopt the dissenter.
- Pay special attention to Juror A (local language specialist) when the content contains local slang, cultural references, or code-switching.
- If the content genuinely contains harmful material, rule "violation" even if all jurors missed it.
- If all jurors are uncertain or their reasoning conflicts beyond resolution, set requires_human_review=true.
- Be specific in your reasoning: cite which juror made which points and evaluate their quality.
"""

BASE_USER_TEMPLATE = """## Original Content

**Source:** {source}
**Country:** {country}
**Detected Language:** {language}

### Content:
```
{content}
```

---

## Juror A Verdict (Local Language Specialist)
**Model:** {model_a}
**Ruling:** {ruling_a}
**Category:** {category_a}
**Confidence:** {confidence_a}

**Reasoning:**
{reasoning_a}

---

## Juror B Verdict (Generalist Reasoner)
**Model:** {model_b}
**Ruling:** {ruling_b}
**Category:** {category_b}
**Confidence:** {confidence_b}

**Reasoning:**
{reasoning_b}

---

## Juror C Verdict (Senior Moderator)
**Model:** {model_c}
**Ruling:** {ruling_c}
**Category:** {category_c}
**Confidence:** {confidence_c}

**Reasoning:**
{reasoning_c}

---

Please deliver your final judgment as JSON:

```json
{{
  "final_verdict": true,
  "category": "hate_speech",
  "confidence": 0.90,
  "adopted_juror": "A",
  "adopted_reason": "Juror A correctly identified the local slur 'XXX' which Jurors B and C missed.",
  "reasoning": "Detailed explanation comparing the jurors' arguments.",
  "requires_human_review": false
}}
```

Output ONLY valid JSON — nothing else."""


# ── Per-country addenda — appended to BASE_SYSTEM ───────────────────
#
# Keep these short and specific. They sit on top of BASE_SYSTEM so we
# don't repeat the categories/rules. Add what's actually distinctive
# about the country's language/legal/cultural context.

_COUNTRY_ADDENDA: dict[str, str] = {
    "SG": """## Country focus: Singapore
- Languages: English, Mandarin/Hokkien, Bahasa Melayu, Tamil — content often code-switches.
- Singlish particles (lah/lor/leh) are normal speech, not violations.
- Racial and religious harmony is legally protected (Sedition Act, MRHA). Be alert to inter-ethnic provocation.
- "CMIO" framing is mainstream; targeted slurs against any of the four groups are violations.""",

    "ID": """## Country focus: Indonesia
- Language: Bahasa Indonesia, with heavy use of regional slang and Javanese/Sundanese loanwords.
- Common slurs: "bangsat", "anjing", "kafir" (against non-Muslims). Their force depends on context.
- Religious blasphemy is a recognised legal category — flag content explicitly attacking Islam, Christianity, Hinduism, or Buddhism as religions.
- Anti-Chinese-Indonesian sentiment ("cina") and LGBTQ-targeting language are sensitive.""",

    "TH": """## Country focus: Thailand
- Language: Thai (no inter-word spaces, particles like "ครับ/ค่ะ" mark register).
- Lèse-majesté (Article 112) makes content insulting the monarchy a serious violation under local law — flag as `political`.
- Common slurs target ethnic minorities (Burmese, Cambodian, Lao migrants).
- Sarcasm and indirect speech are heavy in Thai online culture; surface-clean text can carry strong intent.""",

    "TR": """## Country focus: Turkey
- Language: Turkish (agglutinative — a single word may carry insult + intensifier + person marker).
- Sensitive political topics: Kurdish identity, Armenian genocide recognition, Gülen movement, secularism vs. Islamism.
- Common slurs: "şerefsiz", "orospu çocuğu" — used loosely in slang but cross into hate when targeted.
- Insulting the President is criminalised under Article 299; treat as `political` violation under local law.""",

    "SA": """## Country focus: Saudi Arabia
- Language: Modern Standard Arabic + Gulf dialects (Hejazi, Najdi). Dialect lowers "register" but rarely changes legality.
- Religious content: insults to Islam, prophets, or Sharia are violations under local law (`hate_speech` if targeting Muslims, `political` if targeting the state's religious authority).
- LGBTQ content and explicit sexual content are illegal locally.
- Be sensitive to discussion of the royal family — criticism may be `political` under local norms.""",

    "BR": """## Country focus: Brazil
- Language: Brazilian Portuguese (distinct from European Portuguese — "você" not "tu", "trem" for "thing", etc.).
- Racial slurs: "macaco" (against Black people), "índio" used pejoratively. Football-fan rivalries often carry racial undertones.
- Anti-LGBTQ slurs ("viado", "bicha") are common but context-dependent — reclaimed in some communities.
- Political polarisation (esquerda/direita, "petralha", "bolsonarista") — only flag when it crosses into incitement, not for partisan opinion.""",

    "MX": """## Country focus: Mexico
- Language: Mexican Spanish — heavy slang ("güey", "pendejo", "chingar" derivatives) is everyday speech, not automatic violations.
- Cartel-related content: glorification of CJNG, Sinaloa, etc. → `illegal`. News reporting of cartel activity is not.
- Anti-indigenous slurs ("indio", "naco") and anti-Central-American-migrant language are sensitive.
- Femicide and gender violence are critical topics; misogynistic threats should be flagged as `violence` or `hate_speech`.""",

    "ZA": """## Country focus: South Africa
- 11 official languages; online forums skew English with isiZulu/Afrikaans code-switching.
- Apartheid-era slurs ("k-word" against Black people, racial epithets in Afrikaans) are highly charged and legally prohibited under PEPUDA.
- "Boer", "umlungu", "coconut" — context-dependent, can be reclaimed or used as slurs.
- Xenophobia toward African migrants ("kwerekwere", "makwerekwere") is a recurring violation pattern.""",

    "AE": """## Country focus: United Arab Emirates
- Languages: Arabic + heavy English, Hindi/Urdu, Tagalog from migrant communities.
- Insults targeting Islam, the ruling families, or UAE law are violations under local law (Cybercrime Law).
- LGBTQ content is illegal locally.
- Anti-migrant-worker language (against South Asians, Filipinos) is a frequent hate-speech pattern.""",

    "PH": """## Country focus: Philippines
- Languages: Filipino/Tagalog + English (Taglish is the norm), plus regional languages (Cebuano, Ilocano).
- Common slurs: "bobo", "tanga" — usually casual, not violations unless targeting a group.
- Political polarisation around Marcos/Duterte/Robredo is intense; criticism is not a violation, incitement is.
- Anti-Chinese sentiment ("intsik", related to West Philippine Sea tensions) is rising — flag when targeting ethnic Chinese-Filipinos.""",

    "VN": """## Country focus: Vietnam
- Language: Vietnamese (tonal, diacritics often dropped online — "ko" for "không", "dc" for "được").
- Politically sensitive: criticism of the Party, Article 117/331 prosecutions. Flag overt incitement as `political`; ordinary opinion is not.
- Anti-China sentiment around territorial disputes is widespread; flag explicit ethnic targeting, not state criticism.
- LGBTQ topics are increasingly mainstream; old slurs ("bóng", "pê đê") are used both pejoratively and reclaimed.""",
}


_DEFAULT_PROMPT = JudgePrompt(system=BASE_SYSTEM, user_template=BASE_USER_TEMPLATE)

# Build the registry: country code -> JudgePrompt
_REGISTRY: dict[str, JudgePrompt] = {
    "default": _DEFAULT_PROMPT,
    **{
        code: JudgePrompt(
            system=BASE_SYSTEM + "\n" + addendum,
            user_template=BASE_USER_TEMPLATE,
        )
        for code, addendum in _COUNTRY_ADDENDA.items()
    },
}


def get_prompt(country: Optional[str] = None) -> JudgePrompt:
    """Look up the prompt for a country code (e.g. "ID", "TH").

    Returns the "default" prompt if no country-specific one is registered.
    Lookup is case-insensitive.
    """
    if not country:
        return _REGISTRY["default"]
    return _REGISTRY.get(country.upper(), _REGISTRY["default"])


def register_prompt(country: str, prompt: JudgePrompt) -> None:
    """Add or overwrite the prompt for a country.

    Example:
        register_prompt("MY", JudgePrompt(
            system=BASE_SYSTEM + "\\n## Country focus: Malaysia ...",
            user_template=BASE_USER_TEMPLATE,
        ))
    """
    _REGISTRY[country.upper()] = prompt


def list_countries() -> list[str]:
    """List country codes that have a registered prompt."""
    return sorted(c for c in _REGISTRY.keys() if c != "default")
