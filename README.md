# voting_kit

三席陪审团 + **本地大模型法官**的内容合规判定工具包，**支持 CSV 输入输出**、**按国家自动切换 prompt**。

把三个模型（A 本地专才 / B 开源中坚 / C 云端高阶）对同一段内容的判断包成 `JurorVerdict`，工具包负责裁决——可以纯算法投票，也可以让本地下载好的守卫模型（Qwen3Guard / ShieldGemma / Llama-Guard）当法官。

整个文件夹拷到任何 Python 项目都能跑，**不依赖本仓库其他模块**。

---

## 文件结构

```
voting_kit/
├── __init__.py          # 对外 API
├── schema.py            # 数据模型
├── voting.py            # 阶段一：规则投票
├── prompts.py           # ★ 按国家组织的法官 prompt 注册表（改 prompt 在这里）
├── local_arbiter.py     # 阶段二：本地 HF 模型法官（推荐）
├── llm_arbiter.py       # 阶段二：云端 API 法官
├── decide.py            # 统一入口 decide()
├── csv_runner.py        # ★ CSV 输入输出流水线（run_csv + CLI）
├── jurors_example.py    # ★ jurors_fn 模板，替换成你自己的三个模型
├── example.py           # CLI 单条 demo
├── requirements.txt
└── README.md
```

---

## 安装

```bash
# 本地大模型法官（推荐）
pip install torch transformers accelerate httpx

# 只用纯算法投票则零依赖（Python 3.10+ 即可）
```

---

## 一、CSV 流水线（你最常用的）

你的数据已经是 CSV（`data/cleaned/{COUNTRY}.csv`，列 `source,country,url,title,body,...`），直接用 `run_csv` 跑：

### 步骤 1 — 拷一份 `jurors_example.py`，改成你的三个模型

```bash
cp voting_kit/jurors_example.py my_jurors.py
```

打开 `my_jurors.py`，把三个 stub 替换成真实的模型调用：

```python
from voting_kit import JurorVerdict, ViolationCategory

_LOCAL_PIPE = None

def setup():
    """整个 CSV 处理前调用一次 — 在这里加载本地模型 / API 客户端。"""
    global _LOCAL_PIPE
    from transformers import pipeline
    _LOCAL_PIPE = pipeline("text-classification",
                           model="/data/models/indobert-hate-speech")

def jurors_fn(row, *, content, country, language) -> list[JurorVerdict]:
    """每行 CSV 调一次，返回 [verdict_a, verdict_b, verdict_c]。"""
    cid = row["url"]

    # A: 本地专才
    pred = _LOCAL_PIPE(content[:512])[0]
    a = JurorVerdict(
        content_id=cid, juror="A",
        model_name="indobert-hate-speech",
        violation=pred["label"].lower() == "hate",
        category=ViolationCategory.hate_speech if pred["label"].lower() == "hate" else ViolationCategory.none,
        confidence=pred["score"],
        reasoning=f"Local model: label={pred['label']}, score={pred['score']:.3f}",
        language=language,
    )

    # B: 开源中坚（自己实现，例如调 Together AI / Groq）
    b = call_my_open_source_model(cid, content, language)

    # C: 云端高阶（自己实现，例如调 Claude/GPT/Gemini）
    c = call_my_cloud_model(cid, content, language)

    return [a, b, c]
```

### 步骤 2 — 跑

**命令行（推荐）：**

```bash
python -m voting_kit.csv_runner \
    --input  data/cleaned/ID.csv \
    --output data/results/ID_verdicts.csv \
    --jurors-module my_jurors \
    --mode auto \
    --model-path Qwen/Qwen3Guard-Gen-8B \
    --dtype bfloat16 \
    --limit 100
```

**或者代码里调：**

```python
from voting_kit import run_csv
from my_jurors import jurors_fn, setup

setup()
stats = run_csv(
    input_csv="data/cleaned/ID.csv",
    output_csv="data/results/ID_verdicts.csv",
    jurors_fn=jurors_fn,
    mode="auto",
    model_path="Qwen/Qwen3Guard-Gen-8B",
    dtype="bfloat16",
    limit=100,
)
print(stats)
# {'processed': 100, 'violations': 12, 'clean': 84, 'human_review': 4, ...}
```

### 输入 CSV 列约定

默认读这些列：

| 列 | 用途 | 改名怎么办 |
|----|------|-----------|
| `title` + `body` | 拼成 `content` 喂给法官 | `content_cols=["text"]` |
| `country` | 自动按国家选 prompt | `country_col="market"` |
| `language` | 喂给法官 prompt 的元信息 | `language_col="lang"` |
| `source` | 喂给 prompt（"reddit"/"forum"等） | 没有就空 |

CSV 有 BOM（Excel/Reddit 导出常见）也能直接读，工具里用 `utf-8-sig`。

### 输出 CSV 包含

每行一个最终裁决：

```
content_id, country, language,
final_verdict, category, confidence,
adopted_juror, juror_agreement,
judge_model, requires_human_review,
reasoning, adopted_reason,
juror_a_violation, juror_a_confidence, juror_a_reasoning,
juror_b_violation, juror_b_confidence, juror_b_reasoning,
juror_c_violation, juror_c_confidence, juror_c_reasoning
```

把原 CSV 的额外列复制过来：`extra_input_cols=["url", "subreddit", "created_at"]`。

### 常用选项

```python
run_csv(
    input_csv="...", output_csv="...", jurors_fn=jurors_fn,

    # 模式
    mode="auto",                # vote / weighted / local / api / auto
    model_path="...",           # 本地模型路径或 HF id
    dtype="bfloat16",
    country="ID",               # CSV 没 country 列时的默认值

    # 列映射
    content_cols=["title", "body"],
    country_col="country",
    language_col="language",
    id_col="",                  # 空 → 用 "{文件名}:{行号}" 当 content_id
    extra_input_cols=["url"],

    # 跑数据控制
    limit=0,                    # 0 = 全部
    skip=0,
    resume=True,                # ★ 中断后再跑会自动跳过已完成的行
)
```

**断点续跑**：默认开。再跑一次同样的命令，已经写过 `content_id` 的行会被跳过——大批量跑挂了不丢进度。

---

## 二、怎么改 prompt（按国家定制）

### Prompt 都在 `voting_kit/prompts.py` 里

打开就能看见：

```python
BASE_SYSTEM = """You are the Chief Justice ..."""
BASE_USER_TEMPLATE = """## Original Content ..."""

_COUNTRY_ADDENDA: dict[str, str] = {
    "ID": """## Country focus: Indonesia
- Common slurs: 'bangsat', 'anjing', 'kafir' ...
- Religious blasphemy is a recognised legal category ...
""",
    "TH": """## Country focus: Thailand
- Lèse-majesté (Article 112) ...
""",
    # ... 还有 SG / TR / SA / BR / MX / ZA / AE / PH / VN
}
```

每个国家是一段附录，自动拼到 `BASE_SYSTEM` 后面。法官接收到的最终 system prompt = `BASE_SYSTEM + 该国附录`。

### 三种修改方式

**方式 1：直接改文件**（最简单）

打开 `voting_kit/prompts.py`，编辑：
- 改 `BASE_SYSTEM` → 影响所有国家（违规分类定义、判决规则等）
- 改某国 `_COUNTRY_ADDENDA["ID"]` → 只影响印尼内容
- 加新国家：往 `_COUNTRY_ADDENDA` 里塞一项 `"MY": """## Country focus: Malaysia ..."""`

保存后，调用时传 `country="ID"` 就会用新 prompt。

**方式 2：运行时动态注册**（不动源文件）

```python
from voting_kit import register_prompt, JudgePrompt, BASE_SYSTEM, BASE_USER_TEMPLATE

register_prompt("MY", JudgePrompt(
    system=BASE_SYSTEM + (
        "\n## Country focus: Malaysia\n"
        "- Multilingual: Bahasa Melayu, English, Mandarin, Tamil.\n"
        "- 3R: Race, Religion, Royalty — flag explicit attacks.\n"
        "- 'kafir' / 'bangsa pendatang' in racial framings → hate_speech.\n"
    ),
    user_template=BASE_USER_TEMPLATE,
))

# 之后 country="MY" 就用这个新 prompt
run_csv(..., country="MY")
```

`register_prompt` 用同一个 country code 再调一次会**覆盖**——内置 prompt 不喜欢就这么改。

**方式 3：临时一次性覆盖**

不进注册表，单次调用塞个 `prompt=`：

```python
from voting_kit import JudgePrompt, BASE_USER_TEMPLATE, decide

my_prompt = JudgePrompt(
    system="自己写的整个 system prompt",
    user_template=BASE_USER_TEMPLATE,
)

decide(verdicts, mode="local", model_path="...", prompt=my_prompt)
# 或
run_csv(..., prompt=my_prompt)
```

`prompt=` 参数优先级最高，无视 country 查询。

### 看某国 prompt 长什么样

```bash
python -m voting_kit.example --mode show-prompt --country TH
python -m voting_kit.example --mode show-prompt --country default
```

或代码里：

```python
from voting_kit import get_prompt, list_countries
print(list_countries())             # ['AE','BR','ID','MX','PH','SA','SG','TH','TR','VN','ZA']
print(get_prompt("TH").system)
```

### `BASE_USER_TEMPLATE` 不要随便改

它包含一堆 `{model_a}` / `{ruling_a}` / `{reasoning_a}` 等占位符，**代码里靠这些字段名往里填值**。要改格式可以加东西、调顺序，但占位符必须保留：

```
{source} {country} {language} {content}
{model_a} {ruling_a} {category_a} {confidence_a} {reasoning_a}
{model_b} {ruling_b} {category_b} {confidence_b} {reasoning_b}
{model_c} {ruling_c} {category_c} {confidence_c} {reasoning_c}
```

输出 JSON 的 schema（`final_verdict / category / confidence / adopted_juror / adopted_reason / reasoning / requires_human_review`）也别改 key 名，否则解析失败会进 fallback 路径。

### 内置的 11 国 prompt 写了啥

| 代码 | 国家 | 关键提示内容 |
|------|------|-------------|
| `SG` | 新加坡 | 多语种 + Singlish + 种族宗教和谐法 |
| `ID` | 印尼 | 'bangsat' 等俚语 / 反华 / 宗教敏感 |
| `TH` | 泰国 | 泰语分词 / 王室不敬罪 |
| `TR` | 土耳其 | 库尔德 / 亚美尼亚 / 总统侮辱罪 |
| `SA` | 沙特 | 标准阿语 + 海湾方言 / 宗教 + 王室 |
| `BR` | 巴西 | 巴西葡语 / 'macaco' / 政治极化 |
| `MX` | 墨西哥 | 墨西哥俗语 / 卡特尔 / 反原住民 |
| `ZA` | 南非 | 11 种官方语言 / 隔离遗留侮辱词 |
| `AE` | 阿联酋 | 网络犯罪法 / 反外籍工人 |
| `PH` | 菲律宾 | Taglish / 政治极化 / 反华 |
| `VN` | 越南 | 简化拼写 / Article 117/331 / 反华 |

---

## 三、模式速查

| `mode` | 谁裁决 | 必需参数 | 何时用 |
|--------|--------|---------|--------|
| `"vote"` | 算法多数票 | 无 | 三人一致或 2:1，跑得快又免费 |
| `"weighted"` | 加权投票（A 1.5 / B 1.0 / C 1.2） | 无 | 想让本地专才模型权重更高 |
| `"local"` | **本地** HF 模型 | `model_path` + `content` | 主推：数据不出站、免费、可控 |
| `"api"` | 云端 LLM | `provider` + key + `content` | 不想本地部署 |
| `"auto"` | 先 `vote`，置信度 < 0.7 升级 | 视升级路径 | 性价比最高 |

---

## 四、本地模型法官详解

### 推荐模型

| 模型 | 大小 | 备注 |
|------|------|------|
| `Qwen/Qwen3Guard-Gen-8B` | 8B | 阿里多语言守卫，多语种内容审核首选 |
| `Qwen/Qwen3Guard-Gen-1.7B` | 1.7B | 轻量版，CPU 能跑 |
| `google/shieldgemma-2b` | 2B | Google 守卫，速度快 |
| `google/shieldgemma-9b` | 9B | 更准 |
| `meta-llama/Llama-Guard-3-8B` | 8B | Meta 官方守卫，英文最强 |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | 通用 instruct，多语言不错 |

### 模型路径

`model_path` 既支持本地目录也支持 HF repo id：

```python
model_path = "/data/models/Qwen3Guard-Gen-8B"   # 本地
model_path = "Qwen/Qwen3Guard-Gen-8B"           # HF id（自动下载到 ~/.cache）
```

完全离线流程：

```bash
huggingface-cli download Qwen/Qwen3Guard-Gen-8B --local-dir /data/models/qwen3guard
# 之后调用传本地路径
```

### 调优参数

```python
run_csv(
    ...,
    mode="local",
    model_path="Qwen/Qwen3Guard-Gen-8B",
    dtype="bfloat16",        # auto / bfloat16 / float16 / float32
    device_map="auto",       # 多卡自动切；单卡 "cuda" / "cpu"
    max_new_tokens=1024,
    temperature=0.2,
    trust_remote_code=True,
)
```

模型默认**缓存在内存**——跑完整个 CSV 只加载一次。

释放显存：

```python
from voting_kit import clear_cache
clear_cache()
```

### 直接拿 tokenizer + model 自己控制

```python
from voting_kit import load_local_model

tokenizer, model = load_local_model("Qwen/Qwen3Guard-Gen-8B", dtype="bfloat16")
```

---

## 五、单条 / 程序化使用（不走 CSV）

```python
from voting_kit import JurorVerdict, ViolationCategory, decide

verdicts = [
    JurorVerdict(content_id="post_42", juror="A",
                 model_name="IndoBERT-hate-speech",
                 violation=True, category=ViolationCategory.hate_speech,
                 confidence=0.91, language="id",
                 reasoning="..."),
    JurorVerdict(content_id="post_42", juror="B",
                 model_name="Llama-3.1-70B",
                 violation=False, confidence=0.55, reasoning="..."),
    JurorVerdict(content_id="post_42", juror="C",
                 model_name="claude-haiku-4-5",
                 violation=True, category=ViolationCategory.hate_speech,
                 confidence=0.83, reasoning="..."),
]

final = decide(
    verdicts,
    content="...原帖文本...",
    mode="local",
    model_path="Qwen/Qwen3Guard-Gen-8B",
    country="ID",                 # ← 自动套印尼 prompt
    language="id",
    dtype="bfloat16",
)

print(final.final_verdict, final.category, final.confidence)
```

---

## 公开 API

```python
# 数据模型
JurorVerdict, FinalVerdict, ViolationCategory

# 阶段一：算法
majority_vote(verdicts) -> FinalVerdict | None
weighted_vote(verdicts, weights=None, ...) -> FinalVerdict | None

# 阶段二：本地 LLM 法官
load_local_model(model_path, *, dtype="auto", ...) -> (tokenizer, model)
call_local_arbiter(content_id, content, verdicts, model_path, *,
                   country="", prompt=None, ...) -> FinalVerdict
clear_cache() -> None

# 阶段二：云端 API 法官
call_arbiter(content_id, content, verdicts, *,
             provider="anthropic", country="", prompt=None, ...) -> FinalVerdict

# 统一入口
decide(verdicts, *, mode, content, model_path, country, prompt, ...) -> FinalVerdict

# CSV 流水线
run_csv(input_csv, output_csv, jurors_fn, *, mode, model_path, country,
        content_cols, country_col, ..., resume=True) -> dict

# Prompt 注册表
JudgePrompt(system, user_template)
get_prompt(country) -> JudgePrompt
register_prompt(country, prompt) -> None
list_countries() -> list[str]
BASE_SYSTEM, BASE_USER_TEMPLATE
```

所有 LLM 调用失败都返回 `requires_human_review=True` 的 fallback `FinalVerdict`，错误写在 `adopted_reason`——**不抛异常**，方便批量跑数据。

---

## FAQ

**Q：陪审员 `juror` 字段必须是 `"A" / "B" / "C"` 吗？**
是。`weighted_vote` 用这三个 key 索引权重；改名要传 `weights={...}`。

**Q：CSV 中没有 `country` 列怎么办？**
传 `country="ID"` 给 `run_csv` 当默认值（每行用同一个 prompt）。或者 `country_col=""` 完全禁用。

**Q：跑到一半挂了？**
默认 `resume=True`，重跑同样命令会自动跳过已写入的 `content_id`。

**Q：守卫模型输出不是 JSON？**
工具会先剥 ` ```json ` 围栏、再找第一对 `{...}`。彻底搞不定时返回 fallback verdict 而不是抛异常。某些守卫模型（如原版 ShieldGemma 直接输出 "Yes/No"）可能要自定义 `prompt=JudgePrompt(...)` 把输出引导到 JSON 格式。

**Q：怎么把 `FinalVerdict` 存成 JSON？**
`dataclasses.asdict(final)`；`datetime` 字段记得 `.isoformat()`。

**Q：`run_csv` 的 `jurors_fn` 速度太慢？**
检查模型加载是不是放在 `setup()` 里——而不是每行重新加载一次。

---

## 许可

随主仓库一起。
