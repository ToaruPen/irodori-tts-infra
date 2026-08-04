# Irodori capability-driven voice catalog 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Irodori が active runtime と voice bank の組、動的 voice catalog、caption/emoji 能力、cold-start readiness を所有し、moco がモデル成果物を知らずに fail-closed で合成できる HTTP v1 契約を追加する。

**Architecture:** voice bank manifest から公開用の opaque voice ID と表示 metadata を構築し、`SynthesisPipeline` が generation 条件と voice ID を server-side embedding へ解決する。FastAPI は `GET /capabilities` と安定した error payload だけを公開し、checkpoint、hash、tokenizer、filesystem path は返さない。factory 起動は background load にして `model_loading` を観測可能にするが、既存の固定 `style` preset と caption 非公開という標準経路は変更しない。

**Tech Stack:** Python 3.11+、Pydantic 2、FastAPI、httpx、asyncio、pytest、structlog、`just`

---

## 前提、source of truth、編集境界

- 承認済み設計は moco 側の `/Users/sankenbisha/.codex/worktrees/9540/moco/docs/superpowers/specs/2026-08-04-irodori-v4-dynamic-caption-migration-design.md`。
- `AGENTS.md` は現在、標準経路を v3 600M VoiceDesign、固定 public `style` preset、任意 caption 非公開と定めている。この計画はその規則を維持する。
- 初期 capability は `delivery_caption.supported=false` を返す。`SynthesisRequest` に自由記述 `caption` を追加しない。WebRTC probe が gate を通り、`AGENTS.md` と設計が再承認された後にだけ別計画で追加する。
- `calm`、`cheerful`、`clear` は v4 公式 enum ではなく現行 infra の便宜的 preset である。moco 向け catalog やテストへ固定しない。
- 既存の未コミット変更を保持する。voice bank、checkpoint、service、remote host は変更・再起動・配備しない。
- commit はユーザーが明示した場合だけ作る。本計画には commit step を含めない。
- 旧 moco pin の移行猶予として、既存の `speaker` payload は一時的に受理する。新 client は `voice_id` と `if_generation` を必ず組で送る。
- 話者名、話者数、表示順は runtime data である。テストは生成した fixture catalog から期待値を導出し、実 voice 名、12/13件、manifest 順を固定しない。

## 公開 HTTP 契約

`GET /capabilities` の成功 response は次の形に固定する。`voices` の中身と順序だけが runtime data である。

```json
{
  "contract_version": 1,
  "generation": "opaque-runtime-generation",
  "ready": false,
  "readiness": "model_loading",
  "voices": [
    {
      "id": "opaque-voice-id",
      "label": "表示名",
      "aliases": [],
      "default": true
    }
  ],
  "conditioning": {
    "delivery_caption": {"supported": false, "max_chars": null},
    "emoji": {"supported": true}
  }
}
```

新 client の `POST /synthesize` は次を送る。

```json
{
  "text": "承知しました。",
  "voice_id": "opaque-voice-id",
  "if_generation": "opaque-runtime-generation",
  "num_steps": 40,
  "duration_scale": 1.0,
  "cfg_scale_text": 3.0,
  "cfg_scale_speaker": 5.0
}
```

`voice_id` と `if_generation` は両方指定または両方省略とし、`speaker` と `voice_id` の同時指定を拒否する。省略は旧 client 用互換窓であり、caption なしの legacy speaker 解決を使う。

| HTTP | code | 条件 |
|---:|---|---|
| 404 | `voice_not_found` | `voice_id` または alias が catalog で一意に解決しない |
| 409 | `runtime_generation_mismatch` | `if_generation` と active generation が異なる |
| 503 | `model_not_loaded` | model load または warm-up が完了していない |
| 503 | `voice_bank_invalid` | voice bank または catalog の構造が不正 |

## ファイル構成

- Create `src/irodori_tts_infra/contracts/capabilities.py`: capability response の strict contract。
- Modify `src/irodori_tts_infra/contracts/voices.py`: 旧単体 response を strict `VoiceCapability` へ置換。
- Modify `src/irodori_tts_infra/contracts/synthesis.py`: `voice_id` / `if_generation` の一時互換 validator。
- Modify `src/irodori_tts_infra/contracts/__init__.py`: accepted public contracts の export。
- Modify `src/irodori_tts_infra/voice_bank/models.py`: immutable catalog metadata と一意性 invariant。
- Modify `src/irodori_tts_infra/voice_bank/repository.py`: optional manifest metadata と legacy-safe default。
- Modify `src/irodori_tts_infra/engine/models.py`: `voice_id` / generation を job と pipeline config へ伝搬。
- Modify `src/irodori_tts_infra/engine/errors.py`: typed voice/generation failures。
- Modify `src/irodori_tts_infra/engine/pipeline.py`: generation check と voice ID/alias 解決。
- Create `src/irodori_tts_infra/server/routers/capabilities.py`: thin GET route。
- Modify `src/irodori_tts_infra/server/app.py`, `dependencies.py`, `errors.py`, `main.py`: background readiness と safe error mapping。
- Modify `src/irodori_tts_infra/server/routers/synthesis.py`: v1 fields を job へ渡す。
- Modify `src/irodori_tts_infra/client/async_.py`, `sync.py`: capability client method。
- Modify `src/irodori_tts_infra/config/settings.py`: configured opaque public generation と emoji capability。
- Modify mirrored tests under `tests/contracts/`, `tests/voice_bank/`, `tests/engine/`, `tests/server/`, `tests/client/`, `tests/config/`。
- Modify `.env.example`, `README.md`, `docs/connection.md`: 非秘密設定、contract、readiness、rollback。

### Task 1: strict capability と synthesis request contract

**Files:**
- Create: `tests/contracts/test_capabilities_contracts.py`
- Create: `tests/contracts/test_voice_contracts.py`
- Modify: `tests/contracts/test_synthesis_contracts.py`
- Create: `src/irodori_tts_infra/contracts/capabilities.py`
- Modify: `src/irodori_tts_infra/contracts/voices.py`
- Modify: `src/irodori_tts_infra/contracts/synthesis.py`
- Modify: `src/irodori_tts_infra/contracts/__init__.py`

- [x] **Step 1: production data を固定しない contract tests を RED で追加する**

fixture helper は requested count から値を生成し、実話者名や固定件数を参照しない。

```python
def capability_voices(count: int) -> tuple[VoiceCapability, ...]:
    return tuple(
        VoiceCapability(
            id=f"fixture-voice-{index}",
            label=f"Fixture voice {index}",
            default=index == 0,
        )
        for index in range(count)
    )


@pytest.mark.parametrize("count", [0, 1, 4])
def test_capabilities_preserve_runtime_catalog_without_assuming_names_or_order(count: int) -> None:
    voices = capability_voices(count)
    response = CapabilitiesResponse(
        generation="fixture-generation",
        ready=True,
        readiness="ready",
        voices=voices,
    )

    assert tuple(item.id for item in response.voices) == tuple(item.id for item in voices)
```

併せて unknown field、blank ID/label/generation、`ready` と `readiness` の矛盾、unsupported caption の非 `null` max、重複 alias を拒否する。`SynthesisRequest` は次を検査する。

- legacy `speaker` だけは受理する。
- `voice_id` + `if_generation` は受理する。
- 片方だけ、または `speaker` + `voice_id` は拒否する。
- `caption` は unknown field として拒否する。

- [x] **Step 2: RED を確認する**

Run: `just test tests/contracts/test_capabilities_contracts.py tests/contracts/test_voice_contracts.py tests/contracts/test_synthesis_contracts.py -q`

Expected: capability classes と v1 request fields が存在せず FAIL。

- [x] **Step 3: capability models を最小実装する**

```python
Readiness = Literal["ready", "model_loading", "model_not_loaded", "voice_bank_invalid"]


class DeliveryCaptionCapability(_ContractModel):
    supported: Literal[False] = False
    max_chars: None = None


class EmojiCapability(_ContractModel):
    supported: bool = True


class ConditioningCapabilities(_ContractModel):
    delivery_caption: DeliveryCaptionCapability = DeliveryCaptionCapability()
    emoji: EmojiCapability = EmojiCapability()


class CapabilitiesResponse(_ContractModel):
    contract_version: Literal[1] = 1
    generation: str = Field(min_length=1)
    ready: bool
    readiness: Readiness
    voices: tuple[VoiceCapability, ...]
    conditioning: ConditioningCapabilities = ConditioningCapabilities()
```

`VoiceCapability` は `contracts/voices.py` に置き、unknown field を拒否し、ID・label・alias を strip する。`CapabilitiesResponse` の validator は ID・alias 一意性、alias と別 ID の衝突、default 最大1件、`ready == (readiness == "ready")` を検査する。空 catalog 自体は契約として許し、consumer が明示的に扱う。

`SynthesisRequest` へ `voice_id` と `if_generation` を追加し、model validator で組と排他性を検査する。`SynthesisSegment` の継承にも同じ規則を適用する。

- [x] **Step 4: contract coverage を GREEN にする**

Run: `just test tests/contracts/test_capabilities_contracts.py tests/contracts/test_voice_contracts.py tests/contracts/test_synthesis_contracts.py --cov=irodori_tts_infra.contracts --cov-branch -q`

Expected: 全件 PASS、`contracts/*.py` line/branch 100%。

### Task 2: voice bank manifest を catalog の唯一の所有者にする

**Files:**
- Modify: `src/irodori_tts_infra/voice_bank/models.py`
- Modify: `src/irodori_tts_infra/voice_bank/repository.py`
- Modify: `src/irodori_tts_infra/voice_bank/__init__.py`
- Modify: `tests/voice_bank/test_models.py`
- Modify: `tests/voice_bank/test_repository.py`

- [x] **Step 1: generated manifest fixture で invariant tests を RED にする**

test helper は `count` と index から manifest を作る。期待値は helper の戻り値から導出し、production name/count/order を assert しない。検査対象は次の構造だけとする。

- narrator も通常の catalog entry になる。
- `voice_id`、`label`、`aliases`、`default` を読み込む。
- legacy manifest は narrator に `narrator`、character に manifest key を fallback ID として与える。
- ID 重複、alias ambiguity、alias/ID collision、default 複数を拒否する。
- ref_embed path は catalog response に変換しても現れない。

- [x] **Step 2: RED を確認する**

Run: `just test tests/voice_bank/test_models.py tests/voice_bank/test_repository.py -q`

Expected: catalog model と metadata parser がなく FAIL。

- [x] **Step 3: immutable catalog model と manifest parser を実装する**

```python
@dataclass(frozen=True, slots=True)
class PortableVoice:
    id: str
    label: str
    aliases: tuple[str, ...]
    default: bool
    speaker: SpeakerEmbeddingProfile


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    characters: Mapping[str, CharacterVoice]
    narrator: SpeakerEmbeddingProfile
    catalog: tuple[PortableVoice, ...] = ()
```

repository は `[narrator]` と `[characters.<name>]` の各 table から optional metadata を読む。fallback は既存 manifest の読込互換だけに使い、v4 配備 manifest では全 entry を明示する。

```toml
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"
voice_id = "opaque-narrator-id"
label = "表示名"
aliases = []
default = true
```

`VoiceProfile.__post_init__` は lookup map を入力変更から隔離し、portable voice invariant を検査する。`resolve_voice_id()` は ID または一意 alias から `PortableVoice` を返す。browser 向け変換は path を一切参照しない。

- [x] **Step 4: voice-bank coverage と identifier placement を確認する**

Run:

```bash
just test tests/voice_bank/test_models.py tests/voice_bank/test_repository.py --cov=irodori_tts_infra.voice_bank --cov-branch -q
rg -n "PortableVoice|voice_id|aliases|default" src/irodori_tts_infra/voice_bank tests/voice_bank
```

Expected: 全件 PASS。新 metadata は voice-bank owner とその tests にだけ追加される。

### Task 3: pipeline で generation と voice ID を fail-closed 解決する

**Files:**
- Modify: `src/irodori_tts_infra/engine/errors.py`
- Modify: `src/irodori_tts_infra/engine/models.py`
- Modify: `src/irodori_tts_infra/engine/pipeline.py`
- Modify: `tests/engine/test_models.py`
- Modify: `tests/engine/test_pipeline.py`

- [x] **Step 1: generation/voice resolution の failing tests を追加する**

任意生成した catalog entry を選び、その ID と alias が同じ embedding へ解決されることを検査する。さらに、generation mismatch は backend を一度も呼ばず `RuntimeGenerationMismatchError`、unknown ID は `VoiceNotFoundError` になることを検査する。legacy `speaker` の既存挙動も残す。

- [x] **Step 2: RED を確認する**

Run: `just test tests/engine/test_models.py tests/engine/test_pipeline.py -q`

Expected: new job/config fields と typed errors がなく FAIL。

- [x] **Step 3: job、config、pipeline を最小変更する**

`SynthesisJob` に `voice_id` と `if_generation`、`PipelineConfig` に non-blank `generation` を追加する。`synthesize_job()` は slot acquisition と backend call より前に次の順で検査する。

1. `if_generation` がある場合は exact match を要求する。
2. `voice_id` があれば catalog ID/alias を解決する。
3. なければ一時互換の `speaker`/narrator 解決を使う。
4. 解決した embedding path だけを backend request の `ref_embed` に渡す。

path、hash、model ID を exception message や public details へ含めない。

- [x] **Step 4: engine tests を GREEN にする**

Run: `just test tests/engine/test_models.py tests/engine/test_pipeline.py -q`

Expected: 全件 PASS、mismatch/unknown で backend call count は 0。

### Task 4: capability route、stable errors、sync/async clients

**Files:**
- Create: `src/irodori_tts_infra/server/routers/capabilities.py`
- Modify: `src/irodori_tts_infra/server/routers/synthesis.py`
- Modify: `src/irodori_tts_infra/server/dependencies.py`
- Modify: `src/irodori_tts_infra/server/errors.py`
- Modify: `src/irodori_tts_infra/server/app.py`
- Modify: `src/irodori_tts_infra/client/async_.py`
- Modify: `src/irodori_tts_infra/client/sync.py`
- Fill: `tests/server/routers/test_voices.py`
- Modify: `tests/server/routers/test_synthesis.py`
- Modify: `tests/client/test_async_client.py`
- Modify: `tests/client/test_sync_client.py`

- [x] **Step 1: exact response/error/client tests を RED にする**

`test_voices.py` は空のまま残さず capability router の owner test とする。生成した pipeline catalog を response へ投影し、次を検査する。

- response に `ref_embed`、path separator、generation 以外の runtime secret がない。
- catalog の count/name/order を fixture から導出する。
- sync/async `.capabilities()` が `GET /capabilities` を使う。
- new request の generation mismatch は 409、unknown ID は 404 の `ErrorPayload`。
- legacy request は引き続き成功する。

- [x] **Step 2: RED を確認する**

Run: `just test tests/server/routers/test_voices.py tests/server/routers/test_synthesis.py tests/client/test_async_client.py tests/client/test_sync_client.py -q`

Expected: route、client method、error mapping がなく FAIL。

- [x] **Step 3: thin route と typed mapping を実装する**

`GET /capabilities` は app state の readiness と pipeline catalog を `CapabilitiesResponse` へ変換するだけにする。synthesis router は `_job_from_request()` で new fields を渡す。exception handlers は message を固定した安全な文言にし、内部 exception は structlog へ記録する。

```python
@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities(request: Request) -> CapabilitiesResponse:
    return get_capabilities_response(request)
```

sync/async client は response body を `CapabilitiesResponse.model_validate_json()` で strict validation する。

- [x] **Step 4: route/client tests を GREEN にする**

Run: `just test tests/server/routers/test_voices.py tests/server/routers/test_synthesis.py tests/client/test_async_client.py tests/client/test_sync_client.py -q`

Expected: 全件 PASS。

### Task 5: cold-start readiness を HTTP で観測可能にする

**Files:**
- Modify: `src/irodori_tts_infra/server/app.py`
- Modify: `src/irodori_tts_infra/server/dependencies.py`
- Modify: `src/irodori_tts_infra/server/main.py`
- Modify: `src/irodori_tts_infra/config/settings.py`
- Modify: `tests/server/test_app.py`
- Modify: `tests/server/routers/test_health.py`
- Modify: `tests/config/test_settings.py`

- [x] **Step 1: controlled factory で readiness transition tests を RED にする**

threading event で pipeline factory を止め、`model_loading` response を決定的に観測する。その間の `/synthesize` は 503 `model_not_loaded`、release 後は `ready`、voice-bank 構造失敗は `voice_bank_invalid`、warm-up/backend 失敗は `model_not_loaded` とする。unhandled task exception や path 漏出がないことも検査する。

- [x] **Step 2: RED を確認する**

Run: `just test tests/server/test_app.py tests/server/routers/test_health.py tests/config/test_settings.py -q`

Expected: lifespan が factory 完了まで yield しないため loading state を観測できず FAIL。

- [x] **Step 3: factory load を lifespan-owned background task にする**

factory app の lifespan は `readiness="model_loading"` を設定して直ちに yield し、`asyncio.to_thread(pipeline_factory)` と warm-up を background task で実行する。成功時だけ pipeline と catalog を atomically publish して `ready` にする。shutdown は task を cancel/await し、load 済み backend を一度だけ close する。

直接 `create_app(pipeline)` する unit path は既存どおり deterministic に ready へ遷移させる。voice-bank load 部分だけを typed safe exception へ wrap し、設定・backend の失敗と混同しない。

`IrodoriRuntimeSettings` へ `public_generation: str = "unconfigured"` と
`emoji_conditioning_supported: bool = True` を追加する。`unconfigured` は import と既存unit testを
lightweightに保つ sentinel であり、factory load はこれを `voice_bank_invalid` としてfail closed
する。service設定は起動前に明示的なopaque generationへ更新しなければならない。

`server.main` は settings を一度だけ構築し、generation とemoji capabilityを
`create_app_from_factory()` の初期app stateへ渡してからbackground loadを始める。これにより
`model_loading` 中もgeneration/readinessを返せる。`_build_pipeline(settings)` は同じinstanceを
backendと `PipelineConfig(generation=...)` に渡す。catalogはpipelineがreadyになった時点で
atomically publishするため、loading responseの `voices=[]` を許す。moco側のpollがready responseを
再取得し、固定候補を補わずselectorを更新する。

- [x] **Step 4: readiness tests を GREEN にする**

Run: `just test tests/server/test_app.py tests/server/routers/test_health.py tests/config/test_settings.py -q`

Expected: loading → ready/failed transition が決定的に PASS。

### Task 6: contract E2E と全 catalog iteration

**Files:**
- Create: `tests/integration/test_capability_catalog.py`
- Modify: `pyproject.toml` only if an existing marker declaration must be reused or corrected; do not add a new marker unnecessarily.

- [x] **Step 1: ASGI integration test を runtime-derived iteration で追加する**

test は `/capabilities` を取得し、返された `voices` をそのまま反復して各 `id` と同じ generation で `/synthesize` を呼ぶ。catalog が空なら明示的に fail し、固定 narrator や代替 speaker を挿入しない。各 response は complete RIFF/WAVE、non-empty payload、finite elapsed、readiness 維持を検査する。

- [x] **Step 2: integration test を実行する**

Run: `just test tests/integration/test_capability_catalog.py -q`

Expected: fixture runtime で全 entry PASS。検査対象の name/count/order は test code に存在しない。

### Task 7: 設定・運用文書と非配備 gate

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `docs/connection.md`

- [x] **Step 1: required generation と manifest metadata を文書化する**

文書へ次を明記する。

- `IRODORI_TTS_RUNTIME_PUBLIC_GENERATION` は runtime+voice-bank pair ごとに変更する opaque token。
- browser へ path/hash/tokenizer/checkpoint を公開しない。
- aliases は旧 UI 名の移行データで、曖昧なら server start を fail closed する。
- public caption は未対応で、style preset は server-owned の既存契約。
- service restart、v4 voice bank 置換、標準 generation 変更は別の明示承認が必要。

- [x] **Step 2: repository gate を実行する**

Run:

```bash
just check
git diff --check
rg -n "caption|calm|cheerful|clear" src/irodori_tts_infra/contracts/capabilities.py tests/contracts/test_capabilities_contracts.py
```

Expected: `just check` PASS。capability contract は freeform caption や preset enum を公開せず、`delivery_caption.supported=false` だけを含む。

- [x] **Step 3: dirty worktree と operational boundary を再確認する**

Run: `git status --short`

Expected: 本計画対象の変更と、作業開始前から存在した変更だけが表示される。service、remote host、voice bank、checkpoint には一切変更がない。

## 配備前の別承認 gate

repository 実装が完了しても自動配備しない。次は明示承認後の operational phase でのみ行う。

1. v4 隔離 service を別 port/process/output root で起動する。
2. runtime が返した全 catalog entry を caption なしで反復合成する。
3. WAV、話者同一性、RTF、初回音声、最大 VRAM、health/readiness、終了後 GPU 解放を確認する。
4. 旧 UI 名、とくにアイ、ミウ、narrator の alias 対応を manifest review として人手承認する。対応表をテストへ複製しない。
5. accepted contract をユーザーの明示指示で commit した後にだけ、moco dependency pin の作業へ進む。
6. rollback は承認済み v3 runtime+voice bank+generation を明示的に復元する。別 speaker または v3 へ無通知 fallback しない。

## 完了条件

- `GET /capabilities` が runtime-derived catalog、generation、safe readiness、caption/emoji capability を返す。
- new synthesis request が voice ID と generation を条件付きで解決し、不一致を安定コードで返す。
- cold start の `model_loading` が観測でき、ready 前合成が fail closed する。
- production voice 名、件数、順序を固定する test がない。
- arbitrary caption と v4 公式でない preset が moco-facing contract に入っていない。
- `just check` と `git diff --check` が通る。
- commit、deploy、restart、voice-bank replacement は行われていない。
