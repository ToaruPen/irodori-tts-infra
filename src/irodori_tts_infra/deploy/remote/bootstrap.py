from __future__ import annotations

import os
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import quote

from irodori_tts_infra.deploy.remote._common import _powershell, _ps_quote, _run
from irodori_tts_infra.deploy.remote.sync import (
    resolve_remote_dir,
    resolve_remote_host,
)


def bootstrap_remote(
    *,
    remote_host: str | None = None,
    remote_dir: str | None = None,
    irodori_tts_dir: str | None = None,
    python_version: str | None = None,
    torch_backend_extra: str | None = None,
) -> None:
    host = resolve_remote_host(remote_host)
    directory = resolve_remote_dir(remote_dir)
    irodori_dir = resolve_irodori_tts_dir(irodori_tts_dir)
    runtime_python = resolve_runtime_python(python_version)
    torch_extra = resolve_torch_backend_extra(torch_backend_extra)
    _run(["ssh", host, _powershell(_mkdir_script(directory))])
    _run(
        [
            "ssh",
            host,
            _powershell(
                _bootstrap_script(
                    directory,
                    irodori_tts_dir=irodori_dir,
                    python_version=runtime_python,
                    torch_backend_extra=torch_extra,
                ),
            ),
        ],
    )


def resolve_irodori_tts_dir(value: str | None) -> str:
    directory = value if value is not None else os.getenv("IRODORI_TTS_DIR")
    if directory is None:
        msg = "irodori tts dir is required"
        raise ValueError(msg)
    _reject_blank("irodori_tts_dir", directory)
    return directory


def resolve_runtime_python(value: str | None) -> str:
    python_version = value if value is not None else os.getenv("IRODORI_TTS_RUNTIME_PYTHON", "3.11")
    _reject_blank("python_version", python_version)
    return python_version


def resolve_torch_backend_extra(value: str | None) -> str:
    torch_backend_extra = (
        value if value is not None else os.getenv("IRODORI_TTS_TORCH_BACKEND_EXTRA", "cu128")
    )
    _reject_blank("torch_backend_extra", torch_backend_extra)
    return torch_backend_extra


def _bootstrap_script(
    remote_dir: str,
    *,
    irodori_tts_dir: str,
    python_version: str,
    torch_backend_extra: str,
) -> str:
    runtime_python = "$(Join-Path (Get-Location) '.runtime-venv/Scripts/python.exe')"
    irodori_requirement = (
        f"Irodori-TTS[{torch_backend_extra}] @ {_path_to_file_url(irodori_tts_dir)}"
    )
    return (
        f"Set-Location -LiteralPath {_ps_quote(remote_dir)}; "
        "$lockFile = Join-Path (Get-Location) 'uv.lock'; "
        "if (!(Test-Path -LiteralPath $lockFile)) { "
        "throw 'uv.lock is required; run deploy-sync first' }; "
        "$gitConfig = Join-Path (Get-Location) '.gitconfig-empty'; "
        "if (!(Test-Path -LiteralPath $gitConfig)) { "
        "Set-Content -LiteralPath $gitConfig -Value '' -Encoding UTF8 }; "
        "$env:GIT_CONFIG_GLOBAL = $gitConfig; "
        f"uv venv '.runtime-venv' --python {_ps_quote(python_version)} --clear; "
        f"uv sync --all-extras --locked --python {runtime_python}; "
        f"uv pip install --python {runtime_python} '.[all]'; "
        f"uv pip install --python {runtime_python} {_ps_quote(irodori_requirement)}; "
        f"uv pip check --python {runtime_python}; "
        f"{_runtime_compatibility_check_script(runtime_python)}"
    )


def _runtime_compatibility_check_script(runtime_python: str) -> str:
    return (
        "$compatibilityPath = Join-Path "
        "(Get-Location) '.runtime-compatibility-check.py'; "
        "try { "
        "$compatibilityCode = @'\n"
        f"{_runtime_compatibility_code()}"
        "'@; "
        "Set-Content -LiteralPath $compatibilityPath "
        "-Value $compatibilityCode -Encoding UTF8; "
        f"& {runtime_python} $compatibilityPath; "
        "if ($LASTEXITCODE -ne 0) { "
        "throw 'Irodori-TTS runtime is incompatible with VoiceDesign contract' } "
        "} finally { "
        "Remove-Item -LiteralPath $compatibilityPath "
        "-Force -ErrorAction SilentlyContinue }"
    )


def _runtime_compatibility_code() -> str:
    return (
        "import inspect\n"
        "\n"
        "from irodori_tts.config import ModelConfig\n"
        "from irodori_tts.inference_runtime import SamplingRequest\n"
        "\n"
        "required_sampling_fields = {\n"
        '    "caption",\n'
        '    "ref_embed",\n'
        '    "cfg_scale_caption",\n'
        '    "cfg_scale_speaker",\n'
        '    "cfg_guidance_mode",\n'
        "}\n"
        "required_model_fields = {\n"
        '    "use_caption_condition",\n'
        '    "use_speaker_condition",\n'
        "}\n"
        "sampling_fields = set(inspect.signature(SamplingRequest).parameters)\n"
        "model_fields = set(inspect.signature(ModelConfig).parameters)\n"
        "missing = sorted(\n"
        "    required_sampling_fields.difference(sampling_fields)\n"
        "    | required_model_fields.difference(model_fields)\n"
        ")\n"
        "if missing:\n"
        "    raise RuntimeError(\n"
        '        "Irodori-TTS runtime is incompatible with VoiceDesign contract; "\n'
        "        f\"missing fields: {', '.join(missing)}\"\n"
        "    )\n"
    )


def _mkdir_script(remote_dir: str) -> str:
    return f"New-Item -ItemType Directory -Force -Path {_ps_quote(remote_dir)} | Out-Null"


def _path_to_file_url(path: str) -> str:
    normalized = path.replace("\\", "/")
    if not (PurePosixPath(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute()):
        msg = "irodori_tts_dir must be an absolute path"
        raise ValueError(msg)
    quoted = quote(normalized, safe="/:")
    if normalized.startswith("/"):
        return f"file://{quoted}"
    return f"file:///{quoted}"


def _reject_blank(name: str, value: str) -> None:
    if not value.strip():
        msg = f"{name} must not be blank"
        raise ValueError(msg)
