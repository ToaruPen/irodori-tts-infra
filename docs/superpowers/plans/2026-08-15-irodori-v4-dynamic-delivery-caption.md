# Irodori-TTS v4.1 Dynamic Delivery Caption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Irodori-TTS v4.1 の固定 preset と自由記述 `delivery_caption` を排他的に提供し、Test クライアントと用途別エージェントスキルまで同じ契約で利用できるようにする。

**Architecture:** `SynthesisRequest` が二モードの排他と構造検証を所有し、job と router は値を解釈せず backend まで転送する。backend は自由記述があればそれだけを、なければ固定 preset caption を upstream `SamplingRequest.caption` へ渡す。Test クライアントは同じ fail-fast 規則を実装し、preset と freeform を別 skill に分離する。

**Tech Stack:** Python 3.11+, Pydantic v2, FastAPI, pytest, Ruff, mypy, Irodori-TTS v4.1 Small VoiceDesign, Markdown agent skills

**Repository policy:** この計画ではコミットしない。`irodori-tts-infra/AGENTS.md` と `/Users/sankenbisha/Dev/Test/AGENTS.md` の「commit only when explicitly requested」を優先する。

---

### Task 1: 公開 synthesis contract に二モードを追加する

**Files:**
- Modify: `src/irodori_tts_infra/contracts/synthesis.py`
- Modify: `src/irodori_tts_infra/contracts/__init__.py`
- Test: `tests/contracts/test_synthesis_contracts.py`

- [ ] **Step 1: preset と自由記述の failing tests を追加する**

```python
from irodori_tts_infra.contracts import MAX_DELIVERY_CAPTION_CHARS

@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("alluring", "落ち着いた大人の女性の、艶のある色っぽい声で、自然に話す。"),
        ("lewd", "吐息を交えた大人の女性の、卑猥で挑発的な声で、艶っぽく話す。"),
    ],
)
def test_style_caption_maps_sensual_presets(style: str, expected: str) -> None:
    assert style_caption(style) == expected

def test_synthesis_request_accepts_normalized_delivery_caption() -> None:
    request = SynthesisRequest(text="こんにちは", delivery_caption="  親しい相手へ静かに話す。  ")
    assert request.delivery_caption == "親しい相手へ静かに話す。"
    assert request.style == "neutral"

@pytest.mark.parametrize("value", ["", "   ", "静かに\n話す。", "静かに\t話す。", "静かに\x00話す。"])
def test_synthesis_request_rejects_invalid_delivery_caption(value: str) -> None:
    with pytest.raises(ValidationError, match="delivery_caption"):
        SynthesisRequest(text="こんにちは", delivery_caption=value)

def test_synthesis_request_rejects_overlong_delivery_caption() -> None:
    with pytest.raises(ValidationError, match="delivery_caption"):
        SynthesisRequest(text="こんにちは", delivery_caption="あ" * (MAX_DELIVERY_CAPTION_CHARS + 1))

def test_synthesis_request_rejects_non_neutral_style_with_delivery_caption() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        SynthesisRequest(text="こんにちは", style="alluring", delivery_caption="静かに話す。")
```

- [ ] **Step 2: RED を確認する**

Run: `just test tests/contracts/test_synthesis_contracts.py -q`

Expected: `MAX_DELIVERY_CAPTION_CHARS` または `delivery_caption` が未定義で FAIL。

- [ ] **Step 3: contract を最小実装する**

```python
import unicodedata

MAX_DELIVERY_CAPTION_CHARS = 300
IrodoriStyle = Literal["neutral", "calm", "cheerful", "clear", "alluring", "lewd"]

_STYLE_CAPTIONS = {
    "neutral": None,
    "calm": "穏やかで優しい女性の声で、自然に話す。",
    "cheerful": "明るく親しみやすい女性の声で、自然に話す。",
    "clear": "子どもに伝わるように、ゆっくり明瞭な女性の声で話す。",
    "alluring": "落ち着いた大人の女性の、艶のある色っぽい声で、自然に話す。",
    "lewd": "吐息を交えた大人の女性の、卑猥で挑発的な声で、艶っぽく話す。",
}

class SynthesisRequest(_ContractModel):
    delivery_caption: str | None = None

    @field_validator("delivery_caption")
    @classmethod
    def _normalize_delivery_caption(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("delivery_caption must not be blank")
        if len(stripped) > MAX_DELIVERY_CAPTION_CHARS:
            raise ValueError("delivery_caption must be at most 300 characters")
        if any(unicodedata.category(char) == "Cc" for char in stripped):
            raise ValueError("delivery_caption must not contain control characters")
        return stripped

    @model_validator(mode="after")
    def _validate_voice_selection(self) -> Self:
        # retain existing voice selection checks
        if self.delivery_caption is not None and self.style != "neutral":
            raise ValueError("style and delivery_caption are mutually exclusive")
        return self
```

`contracts/__init__.py` から `MAX_DELIVERY_CAPTION_CHARS` を export する。

- [ ] **Step 4: GREEN を確認する**

Run: `just test tests/contracts/test_synthesis_contracts.py -q`

Expected: PASS。既存の生 `caption` 拒否テストも PASS。

### Task 2: capability contract と server 応答を有効化する

**Files:**
- Modify: `src/irodori_tts_infra/contracts/capabilities.py`
- Modify: `src/irodori_tts_infra/server/dependencies.py`
- Test: `tests/contracts/test_capabilities_contracts.py`
- Test: `tests/server/routers/test_voices.py`

- [ ] **Step 1: supported/max_chars の failing tests を追加する**

```python
from irodori_tts_infra.contracts import DeliveryCaptionCapability, MAX_DELIVERY_CAPTION_CHARS

def test_delivery_caption_capability_requires_limit_exactly_when_supported() -> None:
    supported = DeliveryCaptionCapability(supported=True, max_chars=MAX_DELIVERY_CAPTION_CHARS)
    assert supported.max_chars == 300
    with pytest.raises(ValidationError, match="max_chars"):
        DeliveryCaptionCapability(supported=True, max_chars=None)
    with pytest.raises(ValidationError, match="max_chars"):
        DeliveryCaptionCapability(supported=False, max_chars=300)
```

`test_capabilities_returns_runtime_catalog_without_fixed_names_or_order` の期待値を
`supported is True`、`max_chars == 300` に変える。

- [ ] **Step 2: RED を確認する**

Run: `just test tests/contracts/test_capabilities_contracts.py tests/server/routers/test_voices.py -q`

Expected: `supported=True` が Literal validation で拒否され FAIL。

- [ ] **Step 3: capability と dependency を実装する**

```python
class DeliveryCaptionCapability(_ContractModel):
    supported: bool = False
    max_chars: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_limit(self) -> Self:
        if self.supported != (self.max_chars is not None):
            raise ValueError("max_chars must be present exactly when delivery captions are supported")
        return self
```

`get_capabilities_response()` は次を構築する。

```python
delivery_caption=DeliveryCaptionCapability(
    supported=True,
    max_chars=MAX_DELIVERY_CAPTION_CHARS,
)
```

- [ ] **Step 4: GREEN を確認する**

Run: `just test tests/contracts/test_capabilities_contracts.py tests/server/routers/test_voices.py -q`

Expected: PASS。

### Task 3: 自由記述を router・job・pipeline で欠落なく運ぶ

**Files:**
- Modify: `src/irodori_tts_infra/engine/models.py`
- Modify: `src/irodori_tts_infra/server/routers/synthesis.py`
- Test: `tests/engine/test_pipeline.py`
- Test: `tests/server/routers/test_synthesis.py`

- [ ] **Step 1: 転送の failing tests を追加する**

```python
job = SynthesisJob(
    segment_index=0,
    text="こんにちは",
    ref_embed="speaker.safetensors",
    delivery_caption="親しい相手へ静かに話す。",
)
assert job.to_request().delivery_caption == "親しい相手へ静かに話す。"
```

router test は `/synthesize` へ `delivery_caption` を送り、fake synthesizer の call に同じ値が
残ることを assert する。batch test は segment ごとに preset と自由記述を一件ずつ送る。

- [ ] **Step 2: RED を確認する**

Run: `just test tests/engine/test_pipeline.py tests/server/routers/test_synthesis.py -q`

Expected: `SynthesisJob` が `delivery_caption` を受け取れず FAIL。

- [ ] **Step 3: job と router を実装する**

```python
class SynthesisJob:
    delivery_caption: str | None = None

# SynthesisJob.to_request(...)
delivery_caption=self.delivery_caption,

# _job_from_request(...)
delivery_caption=request.delivery_caption,
```

- [ ] **Step 4: GREEN を確認する**

Run: `just test tests/engine/test_pipeline.py tests/server/routers/test_synthesis.py -q`

Expected: PASS。

### Task 4: backend で preset または自由記述だけを upstream へ渡す

**Files:**
- Modify: `src/irodori_tts_infra/engine/backends/irodori.py`
- Test: `tests/engine/backends/test_irodori.py`
- Test: `tests/gpu/test_phase2_e2e_smoke.py`

- [ ] **Step 1: 自由記述優先の failing test を追加する**

```python
request = resolved_request_factory(
    style="neutral",
    delivery_caption="雨の夜、耳元で囁くように話す。",
)
backend.synthesize(request)
assert runtime.calls[0].caption == "雨の夜、耳元で囁くように話す。"
```

既存の `alluring` preset test では固定 caption が渡ることも assert する。GPU smoke request は
Speaker Inversion と `delivery_caption` を同時指定し、生成結果の WAV contract を維持する。

- [ ] **Step 2: RED を確認する**

Run: `just test tests/engine/backends/test_irodori.py -q`

Expected: runtime call の caption が `None` となり FAIL。

- [ ] **Step 3: backend を実装する**

```python
caption=(
    request.delivery_caption
    if request.delivery_caption is not None
    else style_caption(request.style)
),
```

warm-up は `style_caption(self._settings.warmup_style)` のまま変更しない。

- [ ] **Step 4: GREEN を確認する**

Run: `just test tests/engine/backends/test_irodori.py -q`

Expected: PASS。GPU smoke は既定 gate では実行せず、source/type validation の対象に残す。

### Task 5: infra の source of truth と利用文書を同期する

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/connection.md`
- Modify: `docs/irodori-rvc-architecture.md`

- [ ] **Step 1: 文書の stale assertion を確認する**

Run: `rg -n "arbitrary captions.*not|free-form delivery captions.*unsupported|fixed public style|neutral.*calm.*cheerful.*clear" AGENTS.md README.md docs/connection.md docs/irodori-rvc-architecture.md`

Expected: 固定 preset 限定または自由記述非対応の記述が検出される。

- [ ] **Step 2: 標準経路を二モードへ更新する**

各文書に次を明記する。

```text
- preset mode: style is one of neutral/calm/cheerful/clear/alluring/lewd
- freeform mode: delivery_caption is a validated Japanese delivery direction
- non-neutral style and delivery_caption are mutually exclusive
- raw caption and ref_embed remain private upstream fields
- repository changes do not restart or redeploy the Windows service
```

`docs/connection.md` には `delivery_caption` の JSON 例、`README.md` には二つの短い request 例を載せる。

- [ ] **Step 3: stale assertion が消えたことを確認する**

Run: `rg -n "arbitrary captions.*not|free-form delivery captions.*unsupported" AGENTS.md README.md docs/connection.md docs/irodori-rvc-architecture.md`

Expected: 現行契約を表す一致は 0 件。履歴説明だけが残る場合は「superseded」と明記される。

### Task 6: Test の共通 client contract をTDDで拡張する

**Files:**
- Modify: `/Users/sankenbisha/Dev/Test/tts/tts_engine.py`
- Test: `/Users/sankenbisha/Dev/Test/tts/tests/test_tts_engine.py`

- [ ] **Step 1: single/batch の failing tests を追加する**

```python
def test_synthesize_sends_freeform_delivery_caption(monkeypatch):
    # fake_urlopen captures the JSON payload and returns a valid base64 WAV result
    wav = bare_engine().synthesize(
        text="こんにちは",
        speaker="カスミ",
        delivery_caption="親しい相手へ静かに話す。",
    )
    assert captured["payload"]["delivery_caption"] == "親しい相手へ静かに話す。"
    assert captured["payload"]["style"] == "neutral"
    wav.unlink()

def test_synthesize_rejects_preset_and_freeform_combination() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        bare_engine().synthesize(
            text="こんにちは",
            style="alluring",
            delivery_caption="静かに話す。",
        )
```

batch には `delivery_caption` 許可、blank/301文字/control character/非neutral style併用拒否を追加する。

- [ ] **Step 2: RED を確認する**

Run: `cd /Users/sankenbisha/Dev/Test/tts && uv run --with 'numpy>=1.23.5' --with pytest python -m pytest tests/test_tts_engine.py -q`

Expected: `delivery_caption` 引数または allowed field が未定義で FAIL。

- [ ] **Step 3: client validation と payload を実装する**

```python
MAX_DELIVERY_CAPTION_CHARS = 300
IrodoriStyle = Literal["neutral", "calm", "cheerful", "clear", "alluring", "lewd"]
VALID_STYLES = frozenset({"neutral", "calm", "cheerful", "clear", "alluring", "lewd"})

def _validated_delivery_caption(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("delivery_caption must be a string")
    stripped = value.strip()
    if not stripped or len(stripped) > 300:
        raise ValueError("delivery_caption must contain 1 to 300 characters")
    if any(unicodedata.category(char) == "Cc" for char in stripped):
        raise ValueError("delivery_caption must not contain control characters")
    return stripped
```

`synthesize()` と `stream_batch()` は validation 後、値があるときだけ payload に
`delivery_caption` を含める。`ALLOWED_SEGMENT_FIELDS` に同フィールドを追加する。

- [ ] **Step 4: GREEN を確認する**

Run: `cd /Users/sankenbisha/Dev/Test/tts && uv run --with 'numpy>=1.23.5' --with pytest python -m pytest tests/test_tts_engine.py -q`

Expected: PASS。

### Task 7: Test の single-line CLI と remote helper を二モード化する

**Files:**
- Modify: `/Users/sankenbisha/Dev/Test/tts/say.py`
- Modify: `/Users/sankenbisha/Dev/Test/tts/remote_synth.py`
- Test: `/Users/sankenbisha/Dev/Test/tts/tests/test_say.py`
- Test: `/Users/sankenbisha/Dev/Test/tts/tests/test_remote_synth.py`

- [ ] **Step 1: CLI/helper の failing tests を追加する**

`say.py` に `--delivery-caption "親しい相手へ静かに話す。"` を渡し、fake engine call に
同じ値があることを assert する。`--style alluring --delivery-caption ...` は `SystemExit(2)` と
する。`remote_synth.synthesize()` も `delivery_caption` を engine へ転送する test を加える。

- [ ] **Step 2: RED を確認する**

Run: `cd /Users/sankenbisha/Dev/Test/tts && uv run --with 'numpy>=1.23.5' --with pytest python -m pytest tests/test_say.py tests/test_remote_synth.py -q`

Expected: argparse が未知の option として拒否するか、helper signature error で FAIL。

- [ ] **Step 3: option と転送を実装する**

```python
parser.add_argument("--delivery-caption", default=None, help="自由記述の演技指示（固定styleとは排他）")
if args.delivery_caption is not None and args.style != "neutral":
    parser.error("--style and --delivery-caption are mutually exclusive")
```

`say.py` と `remote_synth.py` は `delivery_caption` を `TTSEngine.synthesize()` へ渡す。

- [ ] **Step 4: GREEN を確認する**

Run: `cd /Users/sankenbisha/Dev/Test/tts && uv run --with 'numpy>=1.23.5' --with pytest python -m pytest tests/test_say.py tests/test_remote_synth.py -q`

Expected: PASS。

### Task 8: read-aloud tag を明示 preset または自由記述へ解決する

**Files:**
- Modify: `/Users/sankenbisha/Dev/Test/tts/read_aloud.py`
- Test: `/Users/sankenbisha/Dev/Test/tts/tests/test_read_aloud.py`

- [ ] **Step 1: tag resolution の failing tests を追加する**

```python
@pytest.mark.parametrize(
    ("direction", "expected_style", "expected_caption"),
    [
        ("", "neutral", None),
        ("style=lewd", "lewd", None),
        ("親しい相手へ耳元で静かに話す。", "neutral", "親しい相手へ耳元で静かに話す。"),
    ],
)
def test_delivery_for_direction(direction, expected_style, expected_caption):
    assert read_aloud.delivery_for_direction(direction) == (expected_style, expected_caption)
```

`style=dramatic` は ValueError。global `--style clear` は全 segment を preset mode にし、
`delivery_caption` を batch へ含めない。自然文 tag は batch に `style=neutral` と
`delivery_caption` を含める。

- [ ] **Step 2: RED を確認する**

Run: `cd /Users/sankenbisha/Dev/Test/tts && uv run --with 'numpy>=1.23.5' --with pytest python -m pytest tests/test_read_aloud.py -q`

Expected: `delivery_for_direction` が存在せず FAIL。

- [ ] **Step 3: heuristic を排他的 resolver へ置換する**

```python
def delivery_for_direction(direction: str) -> tuple[IrodoriStyle, str | None]:
    normalized = direction.strip()
    if not normalized:
        return "neutral", None
    if normalized.startswith("style="):
        style = normalized.removeprefix("style=").strip()
        if style not in VALID_STYLES:
            raise ValueError(f"unknown Irodori style: {style}")
        return cast("IrodoriStyle", style), None
    return "neutral", normalized
```

narration は既存の `calm`、bare dialogue は `neutral`。global override がある場合は全 segment を
override preset とし、それ以外では resolver の tuple から batch dict を構築する。

- [ ] **Step 4: GREEN を確認する**

Run: `cd /Users/sankenbisha/Dev/Test/tts && uv run --with 'numpy>=1.23.5' --with pytest python -m pytest tests/test_read_aloud.py -q`

Expected: PASS。

### Task 9: Test の指示と契約文書を同期する

**Files:**
- Modify: `/Users/sankenbisha/Dev/Test/AGENTS.md`
- Modify: `/Users/sankenbisha/Dev/Test/tts/AGENTS.md`
- Modify: `/Users/sankenbisha/Dev/Test/tts/CLAUDE.md`
- Modify: `/Users/sankenbisha/Dev/Test/.agent/rules/writing_guidelines.md`

- [ ] **Step 1: stale wording を確認する**

Run: `cd /Users/sankenbisha/Dev/Test && rg -n "arbitrary.*caption|fixed.*style|neutral.*calm.*cheerful.*clear|Irodori-TTS v4\b" AGENTS.md tts/AGENTS.md tts/CLAUDE.md .agent/rules/writing_guidelines.md`

Expected: 四 preset 限定、自由記述禁止、v4 表記が検出される。

- [ ] **Step 2: 英語の LLM-facing contract を更新する**

四ファイルへ、v4.1、六 preset、`delivery_caption`、排他規則、三つの tag form を記載する。
writing guideline は自然文 direction を固定 preset に丸めず、そのまま freeform mode へ送ると定める。

- [ ] **Step 3: stale wording の解消を確認する**

Run: `cd /Users/sankenbisha/Dev/Test && rg -n "arbitrary.*caption.*not supported|neutral.*calm.*cheerful.*clear[^\n]*(only|finite)|Irodori-TTS v4 Small" AGENTS.md tts/AGENTS.md tts/CLAUDE.md .agent/rules/writing_guidelines.md`

Expected: 0 matches。

### Task 10: 別の Codex task で skill baseline を実行する

**Files:**
- Read: `/Users/sankenbisha/Dev/Test/.agent/skills/novel_dialogue_playback/SKILL.md`
- Read: `/Users/sankenbisha/Dev/Test/.agent/skills/read_aloud/SKILL.md`
- No persistent eval file: results are retained in the task report

- [ ] **Step 1: fixed checklist を作る**

Median checklist: `[critical] fixed presetを選ぶ`, `[critical] raw captionを送らない`, `正しいsay.py command`。

Edge checklist: `[critical] 自由記述modeを選ぶ`, `[critical] Speaker identity属性をcaptionへ書かない`,
`日本語1〜3文/300文字以内`, `event emojiをtextへ置く`, `presetと併用しない`。

Holdout checklist: `[critical] identityとdeliveryを分離`, `[critical] 矛盾を除去`, `正しいtag/command`。

- [ ] **Step 2: 作成済みの別 task で現行 skill の median と edge を fresh executor に実行させる**

各 executor へ current skill text、scenario、固定 checklist、forbidden behavior、read-only 指示を渡す。

Expected: median は pass 可能、edge は自由記述禁止または preset への丸めにより critical FAIL。

- [ ] **Step 3: baseline failure ledger を記録する**

最低限、`freeform request collapsed to fixed preset`、`identity repeated in caption`、
`localized sound described in caption instead of text emoji` の seen/not-seen を記録する。

### Task 11: preset/freeform skill を分離し静的テストを通す

**Files:**
- Delete: `/Users/sankenbisha/Dev/Test/.agent/skills/novel_dialogue_playback/SKILL.md`
- Delete: `/Users/sankenbisha/Dev/Test/.agent/skills/novel_dialogue_playback/emojis.md`
- Create: `/Users/sankenbisha/Dev/Test/.agent/skills/irodori-preset-speech/SKILL.md`
- Create: `/Users/sankenbisha/Dev/Test/.agent/skills/irodori-freeform-speech/SKILL.md`
- Create: `/Users/sankenbisha/Dev/Test/.agent/skills/irodori-freeform-speech/emojis.md`
- Modify: `/Users/sankenbisha/Dev/Test/.agent/skills/read_aloud/SKILL.md`
- Modify: `/Users/sankenbisha/Dev/Test/.agent/skills/start_tts_server/SKILL.md`
- Test: `/Users/sankenbisha/Dev/Test/tts/tests/test_agent_skills.py`

- [ ] **Step 1: skill structure の failing tests を先に変更する**

`TTS_SKILLS` を `start_tts_server`、`irodori-preset-speech`、`irodori-freeform-speech`、
`read_aloud` の四つへ変更する。preset skill が `--style` と六 preset を含み
`delivery_caption` を含まないこと、freeform skill が `--delivery-caption`、`300`、
`Speaker Inversion`、公式 v4.1 URL を含むこと、read-aloud skill が両 skill 名を参照することを
assert する。emoji vocabulary path は freeform skill 配下へ変える。

- [ ] **Step 2: RED を確認する**

Run: `cd /Users/sankenbisha/Dev/Test/tts && uv run --with pytest python -m pytest tests/test_agent_skills.py -q`

Expected: 新しい skill path が存在せず FAIL。

- [ ] **Step 3: 二つの skill を最小実装する**

`irodori-preset-speech` の frontmatter:

```yaml
---
name: irodori-preset-speech
description: Use when Irodori-TTS speech should use one stable server-owned preset instead of a scene-specific freeform delivery direction.
---
```

`irodori-freeform-speech` の frontmatter:

```yaml
---
name: irodori-freeform-speech
description: Use when Irodori-TTS speech needs a scene-specific freeform delivery direction that no fixed preset captures.
---
```

preset skill は `python3 say.py ... --style <...>`、freeform skill は
`python3 say.py ... --delivery-caption "<Japanese direction>"` の完全な例を一つずつ持つ。
freeform skill は公式順序 `[scene/relationship] [emotion/intensity] [pace/volume/distance/manner]`、
identity再指定禁止、矛盾検査、emoji placement を簡潔に記載する。各 SKILL.md は500語未満にする。

- [ ] **Step 4: GREEN を確認する**

Run: `cd /Users/sankenbisha/Dev/Test/tts && uv run --with pytest python -m pytest tests/test_agent_skills.py -q`

Expected: PASS。

### Task 12: 別の Codex task に skill の empirical tuning を完遂させる

**Files:**
- Modify only if evaluation finds one coherent ambiguity theme: the two new SKILL.md files
- No persistent eval file: structured results remain in final report

- [ ] **Step 1: 初版準備完了を別 task へ通知し、iteration 1 を開始させる**

別 task へ、変更済みskillの絶対path、固定checklist、実行禁止事項、静的test結果を送る。

- [ ] **Step 2: iteration 1 の median と edge を fresh executor で並行実行させる**

Task 10 で固定した checklist を変更せず、各 executor は read-only で artifact、requirement evidence、
phase trace、unclear points、discretionary assumptions、retries、proposed change、tool use を返す。

- [ ] **Step 3: 最小修正を一テーマだけ適用させる**

critical failure または unclear point があれば `Issue / Cause / General Fix Rule` を ledger 化し、
その checklist wording を満たす最小の一テーマだけ skill へ反映する。全項目 pass なら変更しない。

- [ ] **Step 4: iteration 2 を別の fresh executors で実行させる**

Expected: median/edge とも critical failure なし、accuracy 100%、新しい unclear point なし。
条件を満たさなければ Step 3 と fresh iteration を繰り返す。

- [ ] **Step 5: holdout を未使用の fresh executor で実行させる**

Expected: critical failure なし。recent average から15 points以上低下した場合は convergence を撤回し、
scenario edge と skill rule を一テーマだけ修正して median/edge 二連続 clear から再開する。

- [ ] **Step 6: 別 task の差分と評価結果を親 task で検証する**

別 task の報告だけを信用せず、Test repository の skill diff、静的pytest、checklist別の評価根拠を
親 task で読み直す。commit は行わない。

### Task 13: 両 repository を検証する

**Files:**
- Verify all modified files in `/Users/sankenbisha/Dev/irodori-tts-infra`
- Verify all modified files in `/Users/sankenbisha/Dev/Test`

- [ ] **Step 1: infra narrow suites を実行する**

Run: `just test tests/contracts/test_synthesis_contracts.py tests/contracts/test_capabilities_contracts.py tests/engine/test_pipeline.py tests/engine/backends/test_irodori.py tests/server/routers/test_synthesis.py tests/server/routers/test_voices.py -q`

Expected: PASS。

- [ ] **Step 2: infra full gate を実行する**

Run: `just check`

Expected: lint、format、mypy、dead-code、default pytest、coverage がすべて PASS。

- [ ] **Step 3: Test full suite を実行する**

Run: `cd /Users/sankenbisha/Dev/Test/tts && uv run --with 'numpy>=1.23.5' --with pytest --with ruff python -m pytest tests -v`

Expected: 全 test PASS。

- [ ] **Step 4: Test lint/format を実行する**

Run: `cd /Users/sankenbisha/Dev/Test/tts && uv run --with ruff ruff check . && uv run --with ruff ruff format --check .`

Expected: 両 command PASS。

- [ ] **Step 5: repeated identifiers と意図しない差分を確認する**

Run: `rg -n "delivery_caption|alluring|lewd|irodori-preset-speech|irodori-freeform-speech" src tests docs AGENTS.md README.md`

Run: `cd /Users/sankenbisha/Dev/Test && rg -n "delivery_caption|alluring|lewd|irodori-preset-speech|irodori-freeform-speech" tts .agent AGENTS.md`

Expected: identifiers は contract、transport、tests、current docs、skills の意図した owner だけに存在する。

- [ ] **Step 6: git diff と workspace hygiene を確認する**

Run: `git status --short && git diff --check && git diff --stat`

Run: `git -C /Users/sankenbisha/Dev/Test status --short && git -C /Users/sankenbisha/Dev/Test diff --check && git -C /Users/sankenbisha/Dev/Test diff --stat`

Expected: infra は本計画の source/tests/docs、Test は tracked TTS/client/skill/instruction filesだけが変更され、
既存の untracked chat、pics、generated audio、cache、user assets は変更・削除されていない。
