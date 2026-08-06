from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from irodori_tts_infra.config.settings import IrodoriRuntimeSettings
from irodori_tts_infra.engine.backends.irodori import create_irodori_backend
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.engine.models import ResolvedSynthesisRequest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from irodori_tts_infra.contracts.synthesis import IrodoriStyle
    from irodori_tts_infra.engine.backends.irodori import IrodoriBaseBackend

SPEAKER_SUFFIX = ".speaker.safetensors"
DEFAULT_CHECKPOINT = "Aratako/Irodori-TTS-600M-v3-VoiceDesign"
DEFAULT_CHECKPOINT_REVISION = "e863a3a93e652e09afeff3e84823a206a0a60314"
DEFAULT_CHECKPOINT_SHA256 = "93c1f8356857ab4297073f452d01c29015e0db5c83c62109800f8566900f4497"
SEEDS = (1234, 5678)
EVALUATION_CHECKPOINT_STEPS = (1000, 1500, 2000, 2500, 3000)
EVALUATION_TEXT_IDS = (
    "word_unko",
    "word_chinko",
    "word_manko",
    "sentence_unko",
    "sentence_chinko",
    "sentence_manko",
    "control",
)
EVALUATION_STYLES: tuple[Literal["neutral"], Literal["calm"]] = ("neutral", "calm")
MANIFEST_SCHEMA_VERSION = "speaker-checkpoint-evaluation-manifest/v1"
SHA256_HEX_LENGTH = 64
GenerationStatus = Literal["SUCCESS", "ERROR"]


@dataclass(frozen=True, slots=True)
class TextCase:
    text_id: str
    text: str
    control: bool


@dataclass(frozen=True, slots=True)
class CheckpointCandidate:
    model_id: str
    checkpoint_step: int
    speaker_path: Path
    embedding_sha256: str
    training_config_sha256: str
    base_checkpoint: str
    base_checkpoint_sha256: str
    base_revision: str
    run_id: str
    evaluation_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class GenerationCase:
    speaker_path: Path
    text_case: TextCase
    seed: int
    style: IrodoriStyle = "neutral"
    model_id: str | None = None
    checkpoint_step: int | None = None
    checkpoint_candidate: CheckpointCandidate | None = None

    @property
    def speaker(self) -> str:
        if self.model_id is not None:
            return self.model_id
        return _speaker_stem(self.speaker_path)

    @property
    def case_id(self) -> str:
        if self.checkpoint_step is not None:
            return (
                f"{self.speaker}__checkpoint-{self.checkpoint_step}__"
                f"{self.text_case.text_id}__seed-{self.seed}__{self.style}"
            )
        return f"{self.speaker}__{self.text_case.text_id}__seed-{self.seed}__{self.style}"


TEXT_CASES = (
    TextCase(text_id="word_unko", text="うんこ。", control=False),
    TextCase(text_id="word_chinko", text="ちんこ。", control=False),
    TextCase(text_id="word_manko", text="まんこ。", control=False),
    TextCase(
        text_id="sentence_unko",
        text="「うんこ」という言葉を読み上げます。",
        control=False,
    ),
    TextCase(
        text_id="sentence_chinko",
        text="「ちんこ」という言葉を読み上げます。",
        control=False,
    ),
    TextCase(
        text_id="sentence_manko",
        text="「まんこ」という言葉を読み上げます。",
        control=False,
    ),
    TextCase(text_id="control", text="こんにちは。今日はいい天気ですね。", control=True),
)


def build_cases(
    *,
    speaker_paths: Sequence[Path],
    text_cases: Sequence[TextCase],
    seeds: Sequence[int],
    style: IrodoriStyle = "neutral",
) -> tuple[GenerationCase, ...]:
    return tuple(
        GenerationCase(
            speaker_path=speaker_path,
            text_case=text_case,
            seed=seed,
            style=style,
        )
        for speaker_path in sorted(speaker_paths)
        for text_case in text_cases
        for seed in seeds
    )


def build_checkpoint_cases(
    *,
    candidates: Sequence[CheckpointCandidate],
    text_cases: Sequence[TextCase],
    seeds: Sequence[int],
    style: IrodoriStyle = "neutral",
    styles: Sequence[IrodoriStyle] | None = None,
) -> tuple[GenerationCase, ...]:
    return tuple(
        GenerationCase(
            speaker_path=candidate.speaker_path,
            text_case=text_case,
            seed=seed,
            style=selected_style,
            model_id=candidate.model_id,
            checkpoint_step=candidate.checkpoint_step,
            checkpoint_candidate=candidate,
        )
        for candidate in sorted(
            candidates,
            key=lambda candidate: (
                candidate.model_id,
                candidate.checkpoint_step,
                candidate.speaker_path,
            ),
        )
        for text_case in text_cases
        for seed in seeds
        for selected_style in (styles if styles is not None else (style,))
    )


def load_checkpoint_manifest(path: Path) -> tuple[CheckpointCandidate, ...]:
    evaluation_manifest_sha256 = _sha256_file(path)
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        message = f"checkpoint manifest requires schema_version {MANIFEST_SCHEMA_VERSION}"
        raise TypeError(message)
    _validate_evaluation_dimensions(payload)
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        message = "checkpoint manifest models must be a nonempty list"
        raise ValueError(message)
    candidates: list[CheckpointCandidate] = []
    seen_ids: set[tuple[str, int]] = set()
    for raw_model in raw_models:
        candidates.extend(
            _parse_model_candidates(
                raw_model,
                base=path.parent,
                seen_ids=seen_ids,
                evaluation_manifest_sha256=evaluation_manifest_sha256,
            ),
        )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.model_id,
                candidate.checkpoint_step,
                candidate.speaker_path,
            ),
        ),
    )


def _parse_model_candidates(
    raw_model: object,
    *,
    base: Path,
    seen_ids: set[tuple[str, int]],
    evaluation_manifest_sha256: str,
) -> list[CheckpointCandidate]:
    if not isinstance(raw_model, dict):
        message = "checkpoint manifest model entries must be objects"
        raise TypeError(message)
    model_id = _required_string(raw_model, "model_id")
    entries = raw_model.get("checkpoints")
    if not isinstance(entries, list):
        message = f"checkpoint manifest checkpoints must be a list for {model_id}"
        raise TypeError(message)
    steps = tuple(entry.get("checkpoint_step") for entry in entries if isinstance(entry, dict))
    if steps != EVALUATION_CHECKPOINT_STEPS:
        message = (
            f"checkpoint steps for {model_id} must exactly match {EVALUATION_CHECKPOINT_STEPS}"
        )
        raise ValueError(message)
    return [
        _parse_checkpoint_candidate(
            entry,
            model_id=model_id,
            base=base,
            seen_ids=seen_ids,
            evaluation_manifest_sha256=evaluation_manifest_sha256,
        )
        for entry in entries
    ]


def _parse_checkpoint_candidate(
    entry: object,
    *,
    model_id: str,
    base: Path,
    seen_ids: set[tuple[str, int]],
    evaluation_manifest_sha256: str,
) -> CheckpointCandidate:
    if not isinstance(entry, dict):
        message = f"checkpoint entries for {model_id} must be objects"
        raise TypeError(message)
    checkpoint_step = entry.get("checkpoint_step")
    raw_path = entry.get("embedding_path")
    if not isinstance(checkpoint_step, int) or not isinstance(raw_path, str):
        message = (
            f"checkpoint entry for {model_id} requires integer checkpoint_step "
            "and string embedding_path"
        )
        raise TypeError(message)
    candidate_id = (model_id, checkpoint_step)
    if candidate_id in seen_ids:
        message = f"duplicate checkpoint candidate: {model_id} step {checkpoint_step}"
        raise ValueError(message)
    seen_ids.add(candidate_id)
    speaker_path = Path(raw_path)
    speaker_path = (speaker_path if speaker_path.is_absolute() else base / speaker_path).resolve()
    embedding_sha256 = _required_sha256(entry, "embedding_sha256")
    if not speaker_path.is_file():
        message = f"checkpoint embedding does not exist: {speaker_path}"
        raise ValueError(message)
    if _sha256_file(speaker_path) != embedding_sha256:
        message = f"checkpoint embedding SHA-256 mismatch: {speaker_path}"
        raise ValueError(message)
    return CheckpointCandidate(
        model_id=model_id,
        checkpoint_step=checkpoint_step,
        speaker_path=speaker_path,
        embedding_sha256=embedding_sha256,
        training_config_sha256=_required_sha256(entry, "training_config_sha256"),
        base_checkpoint=_required_string(entry, "base_checkpoint"),
        base_checkpoint_sha256=_required_sha256(entry, "base_checkpoint_sha256"),
        base_revision=_required_string(entry, "base_revision"),
        run_id=_required_string(entry, "run_id"),
        evaluation_manifest_sha256=evaluation_manifest_sha256,
    )


def _validate_evaluation_dimensions(payload: dict[str, object]) -> None:
    expected = {
        "text_ids": EVALUATION_TEXT_IDS,
        "seeds": SEEDS,
        "styles": EVALUATION_STYLES,
    }
    for field, expected_values in expected.items():
        value = payload.get(field)
        if not isinstance(value, list) or tuple(value) != expected_values:
            message = f"checkpoint manifest {field} must exactly match {expected_values}"
            raise ValueError(message)


def _required_string(row: dict[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        message = f"checkpoint manifest requires nonempty string {field}"
        raise ValueError(message)
    return value


def _required_sha256(row: dict[str, object], field: str) -> str:
    value = _required_string(row, field)
    if len(value) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        message = f"checkpoint manifest {field} must be a lowercase SHA-256 digest"
        raise ValueError(message)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filter_cases(
    cases: Sequence[GenerationCase],
    *,
    speakers: frozenset[str],
    text_ids: frozenset[str],
    seeds: frozenset[int],
) -> tuple[GenerationCase, ...]:
    return tuple(
        case
        for case in cases
        if (not speakers or case.speaker in speakers)
        and (not text_ids or case.text_case.text_id in text_ids)
        and (not seeds or case.seed in seeds)
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cases, settings = _generation_plan(args)
    if not cases:
        message = "case filters selected no generation cases"
        raise ValueError(message)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    _write_generation_config(output_dir, settings=settings, cases=cases)
    results_path = output_dir / "generation-results.jsonl"

    try:
        backend = create_irodori_backend(settings)
    except (BackendUnavailableError, OSError, RuntimeError, ValueError) as exc:
        _write_backend_failure(results_path, cases=cases, settings=settings, exc=exc)
        print(f"backend creation failed; wrote {len(cases)} ERROR rows to {results_path}")
        return 1

    error_count = _generate_cases(
        backend,
        cases=cases,
        wav_dir=wav_dir,
        results_path=results_path,
        settings=settings,
    )
    print(
        f"generation complete: {len(cases) - error_count} SUCCESS, "
        f"{error_count} ERROR ({results_path})",
    )
    return 1 if error_count else 0


def _generation_plan(
    args: argparse.Namespace,
) -> tuple[tuple[GenerationCase, ...], IrodoriRuntimeSettings]:
    if args.checkpoint_manifest is not None:
        if args.seed or args.text_id or args.speaker or args.style is not None:
            message = "checkpoint evaluation mode does not allow matrix filters"
            raise ValueError(message)
        candidates = load_checkpoint_manifest(args.checkpoint_manifest)
        if not candidates:
            message = f"checkpoint manifest contains no candidates: {args.checkpoint_manifest}"
            raise ValueError(message)
        cases = build_checkpoint_cases(
            candidates=candidates,
            text_cases=TEXT_CASES,
            seeds=SEEDS,
            styles=EVALUATION_STYLES,
        )
        base_contracts = {
            (
                candidate.base_checkpoint,
                candidate.base_revision,
                candidate.base_checkpoint_sha256,
            )
            for candidate in candidates
        }
        if len(base_contracts) != 1:
            message = "checkpoint evaluation candidates must share one base checkpoint contract"
            raise ValueError(message)
        checkpoint, checkpoint_revision, checkpoint_sha256 = next(iter(base_contracts))
        return cases, IrodoriRuntimeSettings(
            checkpoint=checkpoint,
            checkpoint_revision=checkpoint_revision,
            checkpoint_sha256=checkpoint_sha256,
        )
    selected_seeds = tuple(args.seed) if args.seed else SEEDS
    speaker_paths = tuple(sorted(args.speakers_dir.glob(f"*{SPEAKER_SUFFIX}")))
    if not speaker_paths:
        message = f"no {SPEAKER_SUFFIX} files found in {args.speakers_dir}"
        raise ValueError(message)
    all_cases = build_cases(
        speaker_paths=speaker_paths,
        text_cases=TEXT_CASES,
        seeds=selected_seeds,
        style=args.style or "neutral",
    )
    cases = filter_cases(
        all_cases,
        speakers=frozenset(args.speaker),
        text_ids=frozenset(args.text_id),
        seeds=frozenset(selected_seeds),
    )
    return cases, IrodoriRuntimeSettings(
        checkpoint=args.checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        checkpoint_sha256=args.checkpoint_sha256,
    )


def _generate_cases(
    backend: IrodoriBaseBackend,
    *,
    cases: Sequence[GenerationCase],
    wav_dir: Path,
    results_path: Path,
    settings: IrodoriRuntimeSettings,
) -> int:
    error_count = 0
    try:
        with results_path.open("w", encoding="utf-8", newline="\n") as results_file:
            for index, case in enumerate(cases, start=1):
                row = _generate_case(backend, case=case, wav_dir=wav_dir, settings=settings)
                if row["status"] == "ERROR":
                    error_count += 1
                results_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                results_file.flush()
                print(f"[{index}/{len(cases)}] {case.case_id}: {row['status']}")
    finally:
        backend.close()
    return error_count


def _generate_case(
    backend: IrodoriBaseBackend,
    *,
    case: GenerationCase,
    wav_dir: Path,
    settings: IrodoriRuntimeSettings,
) -> dict[str, object]:
    wav_path = wav_dir / f"{case.case_id}.wav"
    started = time.perf_counter()
    try:
        _verify_case_embedding(case)
        audio = backend.synthesize(
            ResolvedSynthesisRequest(
                text=case.text_case.text,
                ref_embed=str(case.speaker_path.resolve()),
                num_steps=settings.num_steps,
                cfg_scale_text=settings.cfg_scale_text,
                cfg_scale_caption=settings.cfg_scale_caption,
                cfg_scale_speaker=settings.cfg_scale_speaker,
                style=case.style,
                seed=case.seed,
                duration_scale=settings.duration_scale,
                num_candidates=settings.num_candidates,
                t_schedule_mode=settings.t_schedule_mode,
                sway_coeff=settings.sway_coeff,
            ),
        )
        wav_path.write_bytes(audio.wav_bytes)
        return _result_row(
            case,
            settings=settings,
            status="SUCCESS",
            elapsed_seconds=time.perf_counter() - started,
            wav_path=wav_path,
        )
    except (BackendUnavailableError, OSError, RuntimeError, ValueError) as exc:
        return _result_row(
            case,
            settings=settings,
            status="ERROR",
            elapsed_seconds=time.perf_counter() - started,
            wav_path=None,
            exc=exc,
        )


def _verify_case_embedding(case: GenerationCase) -> None:
    candidate = case.checkpoint_candidate
    if candidate is not None and _sha256_file(case.speaker_path) != candidate.embedding_sha256:
        message = f"checkpoint embedding SHA-256 mismatch: {case.speaker_path}"
        raise ValueError(message)


def _write_backend_failure(
    results_path: Path,
    *,
    cases: Sequence[GenerationCase],
    settings: IrodoriRuntimeSettings,
    exc: Exception,
) -> None:
    with results_path.open("w", encoding="utf-8", newline="\n") as results_file:
        for case in cases:
            row = _result_row(
                case,
                settings=settings,
                status="ERROR",
                elapsed_seconds=0.0,
                wav_path=None,
                exc=exc,
            )
            results_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _result_row(
    case: GenerationCase,
    *,
    settings: IrodoriRuntimeSettings,
    status: GenerationStatus,
    elapsed_seconds: float,
    wav_path: Path | None,
    exc: Exception | None = None,
) -> dict[str, object]:
    resolved_wav_path = wav_path.resolve() if wav_path is not None else None
    candidate = case.checkpoint_candidate
    row: dict[str, object] = {
        "case_id": case.case_id,
        "speaker": case.speaker,
        "speaker_filename": case.speaker_path.name,
        "text_id": case.text_case.text_id,
        "text": case.text_case.text,
        "control": case.text_case.control,
        "seed": case.seed,
        "style": case.style,
        "checkpoint": candidate.base_checkpoint if candidate is not None else settings.checkpoint,
        "num_steps": settings.num_steps,
        "cfg_scale_text": settings.cfg_scale_text,
        "cfg_scale_caption": settings.cfg_scale_caption,
        "cfg_scale_speaker": settings.cfg_scale_speaker,
        "cfg_guidance_mode": "independent",
        "elapsed_seconds": round(elapsed_seconds, 3),
        "wav_path": str(resolved_wav_path) if resolved_wav_path is not None else None,
        "wav_sha256": (_sha256_file(resolved_wav_path) if resolved_wav_path is not None else None),
        "status": status,
        "exception_type": type(exc).__name__ if exc is not None else None,
        "exception_message": str(exc) if exc is not None else None,
    }
    if candidate is not None:
        row["model_id"] = case.speaker
        row["checkpoint_step"] = candidate.checkpoint_step
        row["embedding_path"] = str(candidate.speaker_path.resolve())
        row["embedding_sha256"] = candidate.embedding_sha256
        row["evaluation_manifest_sha256"] = candidate.evaluation_manifest_sha256
        row["base_checkpoint_sha256"] = candidate.base_checkpoint_sha256
        row["provenance"] = {
            "training_config_sha256": candidate.training_config_sha256,
            "base_checkpoint": candidate.base_checkpoint,
            "base_revision": candidate.base_revision,
            "run_id": candidate.run_id,
        }
    return row


def _write_generation_config(
    output_dir: Path,
    *,
    settings: IrodoriRuntimeSettings,
    cases: Sequence[GenerationCase],
) -> None:
    payload = {
        "checkpoint": settings.checkpoint,
        "num_steps": settings.num_steps,
        "cfg_scale_text": settings.cfg_scale_text,
        "cfg_scale_caption": settings.cfg_scale_caption,
        "cfg_scale_speaker": settings.cfg_scale_speaker,
        "cfg_guidance_mode": "independent",
        "decode_mode": settings.decode_mode,
        "context_kv_cache": settings.context_kv_cache,
        "case_count": len(cases),
        "text_cases": [asdict(text_case) for text_case in TEXT_CASES],
        "seeds": sorted({case.seed for case in cases}),
        "styles": sorted({case.style for case in cases}),
        "speaker_filenames": sorted({case.speaker_path.name for case in cases}),
    }
    (output_dir / "generation-config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _speaker_stem(path: Path) -> str:
    if not path.name.endswith(SPEAKER_SUFFIX):
        message = f"speaker filename must end with {SPEAKER_SUFFIX}: {path.name}"
        raise ValueError(message)
    return path.name[: -len(SPEAKER_SUFFIX)]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--speakers-dir", type=Path)
    source_group.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument(
        "--style",
        choices=("neutral", "calm", "cheerful", "clear"),
    )
    parser.add_argument("--speaker", action="append", default=[])
    parser.add_argument(
        "--text-id",
        choices=tuple(text_case.text_id for text_case in TEXT_CASES),
        action="append",
        default=[],
    )
    parser.add_argument("--seed", type=int, action="append", default=[])
    args = parser.parse_args(argv)
    _validate_checkpoint_args(parser, args)
    return args


def _validate_checkpoint_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    checkpoint_overrides = (
        args.checkpoint,
        args.checkpoint_revision,
        args.checkpoint_sha256,
    )
    if args.checkpoint_manifest is not None:
        if any(value is not None for value in checkpoint_overrides):
            parser.error("checkpoint overrides are not allowed with --checkpoint-manifest")
        return

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
    has_revision = args.checkpoint_revision is not None
    has_sha256 = args.checkpoint_sha256 is not None
    if checkpoint != DEFAULT_CHECKPOINT and not (has_revision and has_sha256):
        parser.error(
            "custom --checkpoint requires both --checkpoint-revision and --checkpoint-sha256"
        )
    if has_revision != has_sha256:
        parser.error("--checkpoint-revision and --checkpoint-sha256 must be provided together")
    args.checkpoint = checkpoint
    args.checkpoint_revision = args.checkpoint_revision or DEFAULT_CHECKPOINT_REVISION
    args.checkpoint_sha256 = args.checkpoint_sha256 or DEFAULT_CHECKPOINT_SHA256


if __name__ == "__main__":
    raise SystemExit(main())
