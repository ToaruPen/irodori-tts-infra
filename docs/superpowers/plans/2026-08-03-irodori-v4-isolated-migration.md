# Irodori-TTS v4 Isolated Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** v3 の既定経路を変えず、将来の学習監視を安定化し、v4-Small の bundled tokenizer を revision/hash 固定で取得できる backend 基盤を作る。

**Architecture:** 学習 supervisor は所有する子 process だけで Python bytecode 生成を無効化し、厳格な upstream source 検査は維持する。backend は Hugging Face の限定 snapshot を commit revision 指定で取得し、model と、設定された場合だけ bundled tokenizer 2 file の SHA-256 を runtime 構築前に検証する。既定設定は v3 600M VoiceDesign のままとし、v4 pin は隔離診断用の文書にだけ置く。

**Tech Stack:** Python 3.11、Pydantic Settings、huggingface-hub、pytest、ruff、Irodori-TTS v4 inference runtime

---

リポジトリ規約により、この計画では commit を作成しない。既存の未コミット変更を保持し、対象 file の現在内容へ最小差分を重ねる。

### Task 1: Detached supervisor の bytecode 自己生成を止める

**Files:**
- Modify: `tests/scripts/test_launch_600m_speaker_training_queue_detached.py`
- Modify: `scripts/launch_600m_speaker_training_queue_detached.py`

- [ ] **Step 1: 子 process 環境の failing assertion を追加する**

`test_supervisor_runs_queue_to_terminal_evidence_and_logs_output` の `popen_calls` 検査へ次を追加する。

```python
    child_env = cast("dict[str, str]", popen_calls[0][1]["env"])
    assert child_env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert child_env["PATH"] == os.environ["PATH"]
```

既存環境を複製して 1 key だけ上書きする契約を確認するため、test 冒頭で `PATH` が存在する
現在の process 環境を利用する。

- [ ] **Step 2: test を実行し RED を確認する**

Run:

```bash
uv run pytest --no-cov tests/scripts/test_launch_600m_speaker_training_queue_detached.py::test_supervisor_runs_queue_to_terminal_evidence_and_logs_output -q
```

Expected: `KeyError: 'env'` で FAIL。現実装が子 process 環境を明示していないためである。

- [ ] **Step 3: queue process に限定した環境を渡す**

`run_supervisor` の `spawn(...)` 直前で次を作り、keyword argument に `env=child_env` を追加する。

```python
        child_env = os.environ.copy()
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        process = spawn(
            command,
            cwd=cast("str", cast("Mapping[str, object]", contract["upstream"])["path"]),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            close_fds=True,
            shell=False,
        )
```

source 検査の `BYTECODE_SUFFIXES` と ignored file 列挙は変更しない。

- [ ] **Step 4: GREEN と既存の source rejection を確認する**

Run:

```bash
uv run pytest --no-cov \
  tests/scripts/test_launch_600m_speaker_training_queue_detached.py::test_supervisor_runs_queue_to_terminal_evidence_and_logs_output \
  tests/scripts/test_launch_600m_speaker_training_queue_detached.py::test_verify_contract_rejects_untracked_source_inside_critical_package \
  -q
```

Expected: 7 tests PASS。

### Task 2: bundled tokenizer pin の設定契約を追加する

**Files:**
- Modify: `tests/config/test_settings.py`
- Modify: `src/irodori_tts_infra/config/settings.py`

- [ ] **Step 1: v3 default と pin pair の failing tests を追加する**

```python
def test_runtime_settings_v3_default_does_not_require_bundled_tokenizer() -> None:
    settings = IrodoriRuntimeSettings()

    assert settings.checkpoint_tokenizer_json_sha256 is None
    assert settings.checkpoint_tokenizer_config_sha256 is None


@pytest.mark.parametrize(
    "values",
    [
        {"checkpoint_tokenizer_json_sha256": "a" * 64},
        {"checkpoint_tokenizer_config_sha256": "b" * 64},
    ],
)
def test_runtime_settings_requires_complete_bundled_tokenizer_pin_pair(
    values: dict[str, str],
) -> None:
    with pytest.raises(ValidationError, match="bundled tokenizer pins"):
        IrodoriRuntimeSettings.model_validate(values)
```

- [ ] **Step 2: test を実行し RED を確認する**

Run:

```bash
uv run pytest --no-cov \
  tests/config/test_settings.py::test_runtime_settings_v3_default_does_not_require_bundled_tokenizer \
  tests/config/test_settings.py::test_runtime_settings_requires_complete_bundled_tokenizer_pin_pair \
  -q
```

Expected: 未定義 field のため FAIL。

- [ ] **Step 3: optional pair と model validator を実装する**

`typing` に `Self`、Pydantic import に `model_validator` を追加し、runtime settings に次を追加する。

```python
    checkpoint_tokenizer_json_sha256: Sha256Hex | None = None
    checkpoint_tokenizer_config_sha256: Sha256Hex | None = None

    @model_validator(mode="after")
    def _require_complete_bundled_tokenizer_pins(self) -> Self:
        json_pin = self.checkpoint_tokenizer_json_sha256
        config_pin = self.checkpoint_tokenizer_config_sha256
        if (json_pin is None) != (config_pin is None):
            msg = "bundled tokenizer pins must be both set or both unset"
            raise ValueError(msg)
        return self
```

- [ ] **Step 4: GREEN と settings suite を確認する**

Run:

```bash
uv run pytest --no-cov tests/config/test_settings.py -q
```

Expected: 全 tests PASS。

### Task 3: revision 固定 snapshot と asset hash 検証を実装する

**Files:**
- Modify: `tests/engine/backends/test_irodori.py`
- Modify: `src/irodori_tts_infra/engine/backends/irodori.py`

- [ ] **Step 1: v3 snapshot 取得の failing test へ変更する**

`test_factory_uses_injected_download_and_runtime_factory` は downloader が file ではなく snapshot
directory を返すようにし、呼出し契約を次で検査する。

```python
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    checkpoint = snapshot / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")

    def snapshot_download_fn(**kwargs: object) -> str:
        download_calls.append(kwargs)
        return str(snapshot)

    backend = create_irodori_backend(
        settings,
        snapshot_download_fn=snapshot_download_fn,
        runtime_factory=runtime_factory,
        runtime_key_cls=FakeRuntimeKey,
        save_wav_fn=fake_save_wav,
        sampling_request_cls=FakeSamplingRequest,
    )

    assert download_calls == [
        {
            "repo_id": "org/model",
            "revision": CUSTOM_REVISION,
            "allow_patterns": ["model.safetensors", "tokenizer/*"],
        }
    ]
    assert runtime_keys[0].checkpoint == str(checkpoint)
```

- [ ] **Step 2: test を実行し RED を確認する**

Run:

```bash
uv run pytest --no-cov tests/engine/backends/test_irodori.py::test_factory_uses_injected_download_and_runtime_factory -q
```

Expected: `snapshot_download_fn` が未定義 keyword のため FAIL。

- [ ] **Step 3: snapshot downloader と checkpoint path 解決を実装する**

`HfHubDownloadFn` を `HfSnapshotDownloadFn` に置換し、factory の injection keyword を
`snapshot_download_fn` にする。download は次の契約にする。

```python
    snapshot_fn = snapshot_download_fn or _import_snapshot_download()
    snapshot_root = Path(
        snapshot_fn(
            repo_id=settings.checkpoint,
            revision=settings.checkpoint_revision,
            allow_patterns=[checkpoint_filename, "tokenizer/*"],
        )
    )
    checkpoint = snapshot_root / checkpoint_filename
    _verify_file_sha256(
        checkpoint,
        expected=settings.checkpoint_sha256,
        label="checkpoint",
    )
```

import helper は `huggingface_hub.snapshot_download` を遅延 import する。

```python
def _import_snapshot_download() -> HfSnapshotDownloadFn:
    try:
        module = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise BackendUnavailableError(INSTALL_HINT) from exc
    return cast("HfSnapshotDownloadFn", module.snapshot_download)
```

既存の injected downloader call は同じ test file 内ですべて `snapshot_download_fn` へ移し、
fixture は checkpoint を含む directory を返すようにする。

- [ ] **Step 4: v3 snapshot test の GREEN を確認する**

Run:

```bash
uv run pytest --no-cov tests/engine/backends/test_irodori.py::test_factory_uses_injected_download_and_runtime_factory -q
```

Expected: PASS。

- [ ] **Step 5: bundled tokenizer 不在・不一致の failing tests を追加する**

```python
@pytest.mark.parametrize("failure", ["missing", "mismatch"])
def test_factory_rejects_invalid_bundled_tokenizer_before_runtime_creation(
    tmp_path: Path,
    failure: str,
) -> None:
    snapshot = tmp_path / "snapshot"
    tokenizer_dir = snapshot / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    checkpoint = snapshot / "model.safetensors"
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_config = tokenizer_dir / "tokenizer_config.json"
    checkpoint.write_bytes(b"v4 checkpoint")
    tokenizer_json.write_bytes(b"v4 tokenizer")
    tokenizer_config.write_bytes(b"v4 tokenizer config")
    tokenizer_json_sha256 = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
    settings = runtime_settings(
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_tokenizer_json_sha256=tokenizer_json_sha256,
        checkpoint_tokenizer_config_sha256=hashlib.sha256(
            tokenizer_config.read_bytes()
        ).hexdigest(),
    )
    runtime_keys: list[FakeRuntimeKey] = []
    if failure == "missing":
        tokenizer_json.unlink()
        expected = "bundled tokenizer.json is missing or unreadable"
    else:
        tokenizer_json.write_bytes(b"tampered tokenizer")
        expected = "bundled tokenizer.json SHA-256 mismatch"

    with pytest.raises(BackendUnavailableError, match=expected):
        create_irodori_backend(
            settings,
            snapshot_download_fn=lambda **_kwargs: str(snapshot),
            runtime_factory=lambda key: runtime_keys.append(cast("FakeRuntimeKey", key))
            or FakeRuntime(),
            runtime_key_cls=FakeRuntimeKey,
            save_wav_fn=fake_save_wav,
            sampling_request_cls=FakeSamplingRequest,
        )

    assert runtime_keys == []
```

- [ ] **Step 6: test を実行し RED を確認する**

Run:

```bash
uv run pytest --no-cov tests/engine/backends/test_irodori.py::test_factory_rejects_invalid_bundled_tokenizer_before_runtime_creation -q
```

Expected: 例外が発生せず `DID NOT RAISE` で 2 cases とも FAIL。現実装が tokenizer pin を
消費していないためである。

- [ ] **Step 7: 共通 file verifier と tokenizer 検証を実装する**

```python
def _verify_file_sha256(path: Path, *, expected: str, label: str) -> None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        msg = f"{label} is missing or unreadable: {path}"
        raise BackendUnavailableError(msg) from exc
    actual = digest.hexdigest()
    if actual != expected:
        msg = f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        raise BackendUnavailableError(msg)


def _verify_bundled_tokenizer(snapshot_root: Path, settings: IrodoriRuntimeSettings) -> None:
    json_pin = settings.checkpoint_tokenizer_json_sha256
    config_pin = settings.checkpoint_tokenizer_config_sha256
    if json_pin is None or config_pin is None:
        return
    _verify_file_sha256(
        snapshot_root / "tokenizer" / "tokenizer.json",
        expected=json_pin,
        label="bundled tokenizer.json",
    )
    _verify_file_sha256(
        snapshot_root / "tokenizer" / "tokenizer_config.json",
        expected=config_pin,
        label="bundled tokenizer_config.json",
    )
```

checkpoint 検証の直後、runtime key 作成の前に `_verify_bundled_tokenizer` を呼ぶ。

- [ ] **Step 8: bundled tokenizer tests を GREEN にする**

Run:

```bash
uv run pytest --no-cov tests/engine/backends/test_irodori.py::test_factory_rejects_invalid_bundled_tokenizer_before_runtime_creation -q
```

Expected: 2 tests PASS。既存の v3 factory success test が tokenizer pin 未指定の snapshot を
引き続き受理するため、optional fallback も同時に保たれる。

- [ ] **Step 9: backend suite を GREEN にする**

Run:

```bash
uv run pytest --no-cov tests/engine/backends/test_irodori.py -q
```

Expected: 全 tests PASS。

### Task 4: v4 非配備診断 pin を文書化する

**Files:**
- Create: `docs/deploy/irodori-v4-diagnostic.env.example`
- Modify: `tests/deploy/test_remote.py`
- Modify: `docs/superpowers/specs/2026-08-03-irodori-v4-isolated-migration-design.md`

- [ ] **Step 1: 診断 file の failing contract test を追加する**

```python
def test_v4_diagnostic_env_is_pinned_and_does_not_replace_v3_default() -> None:
    diagnostic = Path("docs/deploy/irodori-v4-diagnostic.env.example").read_text(
        encoding="utf-8"
    )
    default_env = Path(".env.example").read_text(encoding="utf-8")

    assert "IRODORI_TTS_RUNTIME_CHECKPOINT=Aratako/Irodori-TTS-v4-Small" in diagnostic
    assert "IRODORI_TTS_RUNTIME_CHECKPOINT_REVISION=e4aaac4" in diagnostic
    assert "IRODORI_TTS_RUNTIME_CHECKPOINT_SHA256=5863c986" in diagnostic
    assert "IRODORI_TTS_RUNTIME_CHECKPOINT_TOKENIZER_JSON_SHA256=6a0734cf" in diagnostic
    assert "IRODORI_TTS_RUNTIME_CHECKPOINT_TOKENIZER_CONFIG_SHA256=d229a271" in diagnostic
    assert "IRODORI_TTS_RUNTIME_CHECKPOINT=Aratako/Irodori-TTS-600M-v3-VoiceDesign" in default_env
```

実際の assertion では省略 prefix ではなく 40/64 文字の完全な pin を使う。

- [ ] **Step 2: test を実行し RED を確認する**

Run:

```bash
uv run pytest --no-cov tests/deploy/test_remote.py::test_v4_diagnostic_env_is_pinned_and_does_not_replace_v3_default -q
```

Expected: 診断 file 不在で FAIL。

- [ ] **Step 3: 完全 pin を持つ診断 env example を作る**

```dotenv
# Isolated non-deployment Irodori-TTS v4 diagnostic only.
IRODORI_TTS_RUNTIME_CHECKPOINT=Aratako/Irodori-TTS-v4-Small
IRODORI_TTS_RUNTIME_CHECKPOINT_REVISION=e4aaac4df355ff560dcd35e0dae272c3a759317b
IRODORI_TTS_RUNTIME_CHECKPOINT_SHA256=5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593
IRODORI_TTS_RUNTIME_CHECKPOINT_TOKENIZER_JSON_SHA256=6a0734cf21c802169defaffe719bc2ef12bb9d0be37e54b61ed27aa89394723d
IRODORI_TTS_RUNTIME_CHECKPOINT_TOKENIZER_CONFIG_SHA256=d229a271c64de1a7939d20d3665498e873fa91d5ee2edf135d73ec752cb9c9d3
```

設計仕様にも tokenizer 2 file の完全 hash を追記する。

- [ ] **Step 4: GREEN を確認する**

Run:

```bash
uv run pytest --no-cov tests/deploy/test_remote.py::test_v4_diagnostic_env_is_pinned_and_does_not_replace_v3_default -q
```

Expected: PASS。

### Task 5: repository 回帰検証

**Files:**
- Verify only; no additional files.

- [ ] **Step 1: 新規識別子の ownership を確認する**

Run:

```bash
rg -n "PYTHONDONTWRITEBYTECODE|checkpoint_tokenizer_json_sha256|checkpoint_tokenizer_config_sha256|snapshot_download_fn" \
  scripts src tests docs
```

Expected: supervisor、runtime settings、Irodori backend、その直接 tests と v4 診断文書だけに現れる。

- [ ] **Step 2: 対象 suite を実行する**

Run:

```bash
uv run pytest --no-cov \
  tests/scripts/test_launch_600m_speaker_training_queue_detached.py \
  tests/config/test_settings.py \
  tests/engine/backends/test_irodori.py \
  tests/deploy/test_remote.py \
  -q
```

Expected: 全 tests PASS。

- [ ] **Step 3: static checks を実行する**

Run:

```bash
uv run ruff check \
  scripts/launch_600m_speaker_training_queue_detached.py \
  tests/scripts/test_launch_600m_speaker_training_queue_detached.py \
  src/irodori_tts_infra/config/settings.py \
  tests/config/test_settings.py \
  src/irodori_tts_infra/engine/backends/irodori.py \
  tests/engine/backends/test_irodori.py \
  tests/deploy/test_remote.py
uv run ruff format --check \
  scripts/launch_600m_speaker_training_queue_detached.py \
  tests/scripts/test_launch_600m_speaker_training_queue_detached.py \
  src/irodori_tts_infra/config/settings.py \
  tests/config/test_settings.py \
  src/irodori_tts_infra/engine/backends/irodori.py \
  tests/engine/backends/test_irodori.py \
  tests/deploy/test_remote.py
uv run mypy
```

Expected: PASS。既存の unrelated dirty file に起因する failure があれば対象と出力を記録し、
この変更で導入した failure だけを修正する。

- [ ] **Step 4: full local gate を実行する**

Run:

```bash
just check
```

Expected: PASS、または GPU・network・SSH を必要とする marker は default test から除外される。

### Task 6: v4 GPU 作業への非配備 handoff

**Files:**
- Operational only; runtime asset や生成音声は repository に書かない。

- [ ] **Step 1: 現行 v3 checkout を変更せず v4 checkout を作る**

Windows GPU host 上で別 directory に upstream commit
`8ca3acb58ab4e19ad6d594aaed6bafe3e88f7f71` を detached checkout する。既存の
`C:\Users\takut\Dev\Irodori-TTS` は変更しない。

- [ ] **Step 2: v4 runtime と v3 checkpoint の後方互換 smoke を実行する**

v4 checkout の runtime から現行 v3 checkpoint と 1 個の v3 Speaker Inversion embedding を
使い、neutral と calm を固定本文・seed で生成する。非空 WAV、有限 sample、例外なし、終了後
training process なしを証跡化する。

- [ ] **Step 3: v4 model と v3 embedding の診断を実行する**

`docs/deploy/irodori-v4-diagnostic.env.example` の pin で v4 snapshot を取得し、同じ 1 話者・
同じ case を生成する。この結果は診断限定と明記し、voice bank を更新しない。

- [ ] **Step 4: v4 Speaker Inversion pilot の開始可否を判定する**

runtime 後方互換、v4 model load、bundled tokenizer hash、GPU closure がすべて通った場合だけ、
fresh initialization の 1 話者 pilot を別 versioned output root で開始する。不合格の場合は v3
標準経路を維持し、失敗証跡を報告する。
