"""Stage 2 TTS adapter, selectable through ``tts.engine``."""

import atexit
import json
import logging
import subprocess
import threading
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .config import VALID_ENGINES
from .validate import NormalizedTurn

logger = logging.getLogger(__name__)


def get_engine(config: dict[str, Any]) -> str:
    engine = config["tts"].get("engine", "edge_tts")
    if engine not in VALID_ENGINES:
        raise ValueError(
            f"Unsupported tts.engine: {engine} (available: {sorted(VALID_ENGINES)})"
        )
    return engine


def _project_root() -> Path:
    """Repository root, two levels above this file (src/tts5703/tts_engine.py)."""
    return Path(__file__).resolve().parents[2]


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def turn_audio_extension(config: dict[str, Any]) -> str:
    return ".mp3" if get_engine(config) == "edge_tts" else ".wav"


_EDGE_TTS_SEMANTIC_RATES = {
    "slow": "-20%",
    "normal": "+0%",
    "fast": "+20%",
}
_KOKORO_SEMANTIC_RATES = {
    "slow": 0.8,
    "normal": 1.0,
    "fast": 1.2,
}
_COSYVOICE_SEMANTIC_RATES = {
    "slow": 0.8,
    "normal": 1.0,
    "fast": 1.2,
}
_COSYVOICE_AROUSAL_INSTRUCTIONS = {
    "low": "Use a calm, soft, subdued delivery.",
    "medium": "Use a neutral, moderately expressive delivery.",
    "high": "Use an energetic, intense delivery.",
}
_COSYVOICE_AFFECT_INSTRUCTIONS = {
    "neutral": "Use a neutral, composed tone.",
    "distressed": "Use a distressed, worried, and sad tone.",
}
_COSYVOICE_INSTRUCTION_PREFIX = "You are a helpful assistant."
_COSYVOICE_END_OF_PROMPT = "<|endofprompt|>"


def rate_to_edge_tts(rate: str) -> str:
    """Map semantic rates to EdgeTTS percentages, preserving legacy values."""
    return _EDGE_TTS_SEMANTIC_RATES.get(rate, rate)


def rate_to_kokoro_speed(rate: str) -> float:
    """Map semantic or legacy percentage rates to a Kokoro speed multiplier."""
    if rate in _KOKORO_SEMANTIC_RATES:
        return _KOKORO_SEMANTIC_RATES[rate]
    return max(0.1, 1 + int(rate[:-1]) / 100)


def _rate_to_kokoro_speed(rate: str) -> float:
    """Backward-compatible alias for the pre-v0.2 internal helper."""
    return rate_to_kokoro_speed(rate)


def rate_to_cosyvoice_speed(rate: str) -> float:
    """Map semantic or legacy percentage rates to CosyVoice's speed argument."""
    if rate in _COSYVOICE_SEMANTIC_RATES:
        return _COSYVOICE_SEMANTIC_RATES[rate]
    return max(0.1, 1 + int(rate[:-1]) / 100)


class BackendControlError(ValueError):
    """Raised when a schema-valid control has no mapping in the selected backend.

    Canonical schema v0.2 keeps ``coarse_affect`` an open string so corpus design
    is not constrained by one backend's vocabulary. This error is the
    backend-specific counterpart: schema-valid intent the selected engine cannot
    currently translate.
    """


def _check_cosyvoice_mapping(
    field: str, value: str | None, mapping: dict[str, str]
) -> None:
    if value is None or value in mapping:
        return
    raise BackendControlError(
        f"Unsupported CosyVoice {field} mapping: {value!r}. "
        f"Currently supported mappings: {', '.join(mapping)}."
    )


def preflight_cosyvoice_controls(turn: NormalizedTurn) -> None:
    """Reject requested controls CosyVoice cannot map, before synthesis starts."""
    _check_cosyvoice_mapping("arousal", turn.arousal, _COSYVOICE_AROUSAL_INSTRUCTIONS)
    _check_cosyvoice_mapping(
        "coarse_affect", turn.coarse_affect, _COSYVOICE_AFFECT_INSTRUCTIONS
    )


def preflight_backend_controls(turn: NormalizedTurn, config: dict[str, Any]) -> None:
    """Validate one turn against the selected backend's control mappings.

    Backends that ignore a control (see ``engine_capabilities``) must not reject
    it here; the requested value is preserved and reported as ignored instead.
    """
    if get_engine(config) == "cosyvoice":
        preflight_cosyvoice_controls(turn)


def preflight_dialogue_controls(
    turns: list[NormalizedTurn], config: dict[str, Any]
) -> None:
    """Fail a whole dialogue before spending any synthesis time on it."""
    for turn in turns:
        try:
            preflight_backend_controls(turn, config)
        except BackendControlError as error:
            raise BackendControlError(f"turn {turn.turn_id}: {error}") from error


def build_cosyvoice_instruction(
    arousal: str | None, coarse_affect: str | None
) -> str | None:
    """Build one provisional CosyVoice3 instruction from requested controls."""
    _check_cosyvoice_mapping("arousal", arousal, _COSYVOICE_AROUSAL_INSTRUCTIONS)
    _check_cosyvoice_mapping(
        "coarse_affect", coarse_affect, _COSYVOICE_AFFECT_INSTRUCTIONS
    )
    controls = [
        instruction
        for instruction in (
            _COSYVOICE_AROUSAL_INSTRUCTIONS.get(arousal),
            _COSYVOICE_AFFECT_INSTRUCTIONS.get(coarse_affect),
        )
        if instruction is not None
    ]
    if not controls:
        return None
    return f"{_COSYVOICE_INSTRUCTION_PREFIX} {' '.join(controls)}{_COSYVOICE_END_OF_PROMPT}"


def build_cosyvoice_request(
    turn: NormalizedTurn,
    prompt_text: str,
    prompt_wav: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Translate corpus controls into the isolated CosyVoice worker protocol."""
    instruction = build_cosyvoice_instruction(turn.arousal, turn.coarse_affect)
    request: dict[str, Any] = {
        "text": turn.text,
        "prompt_text": prompt_text,
        "prompt_wav": str(prompt_wav),
        "output_path": str(output_path),
        "speed": rate_to_cosyvoice_speed(turn.rate),
        "mode": "instruct2" if instruction is not None else "zero_shot",
    }
    if instruction is not None:
        request["instruction"] = instruction
    return request


@lru_cache(maxsize=4)
def _get_kokoro_pipeline(lang_code: str, device: str | None):
    # Lazy import keeps EdgeTTS runs independent of Kokoro model loading.
    from kokoro import KPipeline

    return KPipeline(lang_code=lang_code, device=device)


_COSYVOICE_WORKER_SCRIPT = Path(__file__).resolve().parent / "cosyvoice_worker.py"
_COSYVOICE_TERMINATE_TIMEOUT_SEC = 5
# Official output rate of Fun-CosyVoice3-0.5B. This is the expected rate, not an
# observed one: the worker reports its own rate at startup, but that value is not
# threaded back through synthesize_turn.
_COSYVOICE3_EXPECTED_SAMPLE_RATE = 24_000
_SAMPLE_RATE_NOT_RUNTIME_VERIFIED = "not_runtime_verified"


def _drain_cosyvoice_stderr(proc: subprocess.Popen[str]) -> deque[str]:
    """Drain worker diagnostics so its stderr pipe cannot block a long batch."""
    tail: deque[str] = deque(maxlen=50)

    def drain() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            message = line.rstrip()
            if message:
                tail.append(message)
                logger.debug("event=cosyvoice_worker_stderr message=%s", message)

    threading.Thread(
        target=drain,
        name="cosyvoice-stderr",
        daemon=True,
    ).start()
    return tail


def _cosyvoice_worker_error(
    proc: subprocess.Popen[str], stderr_tail: deque[str], fallback: str
) -> str:
    """Return the most useful available diagnostic from a failed worker."""
    if proc.poll() is None:
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    return "\n".join(stderr_tail) or fallback


@lru_cache(maxsize=1)
def _get_cosyvoice_worker(
    python_bin: str,
    repo_dir: str,
    model_dir: str,
    load_trt: bool,
    load_vllm: bool,
    fp16: bool,
) -> subprocess.Popen[str]:
    """Start (once) and reuse the CosyVoice worker subprocess for this run.

    CosyVoice3's dependencies live in their own virtual environment,
    separate from this project's .venv (see ``tts.cosyvoice.python_bin``),
    so synthesis happens in a subprocess speaking JSON-lines over stdio
    rather than an in-process import.
    """
    proc = subprocess.Popen(
        [python_bin, str(_COSYVOICE_WORKER_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stderr_tail = _drain_cosyvoice_stderr(proc)
    proc._cosyvoice_stderr_tail = stderr_tail
    atexit.register(_shutdown_cosyvoice_worker, proc)

    assert proc.stdin is not None and proc.stdout is not None
    init = {
        "repo_dir": repo_dir,
        "model_dir": model_dir,
        "load_trt": load_trt,
        "load_vllm": load_vllm,
        "fp16": fp16,
    }
    logger.debug(
        "event=cosyvoice_worker_start python_bin=%s model_dir=%s", python_bin, model_dir
    )
    proc.stdin.write(json.dumps(init) + "\n")
    proc.stdin.flush()
    response_line = proc.stdout.readline()
    try:
        response = json.loads(response_line) if response_line else {}
    except json.JSONDecodeError as error:
        proc.kill()
        raise RuntimeError(
            "CosyVoice worker returned invalid startup data: "
            f"{response_line.rstrip()!r}"
        ) from error
    if response.get("status") != "ready":
        message = response.get("message") or _cosyvoice_worker_error(
            proc, stderr_tail, f"worker exited with code {proc.poll()}"
        )
        if proc.poll() is None:
            proc.kill()
        raise RuntimeError(
            f"Failed to start CosyVoice worker (python_bin={python_bin}): {message}"
        )
    logger.debug("event=cosyvoice_worker_ready model_dir=%s", model_dir)
    return proc


def _shutdown_cosyvoice_worker(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
        except Exception:  # noqa: BLE001, S110 - best-effort interpreter cleanup
            pass


def _terminate_cosyvoice_worker(proc: subprocess.Popen[str]) -> None:
    """Terminate and reap a worker whose protocol stream is no longer usable.

    Without this, a worker that stopped answering but is still alive keeps its
    GPU memory for the rest of the batch, because the cached handle is dropped
    and nothing else ever waits on the process.
    """
    if proc.poll() is not None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=_COSYVOICE_TERMINATE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            # The worker exited between the timeout and the kill; it still needs
            # reaping below, so this race is not a cleanup failure.
            logger.debug(
                "event=cosyvoice_worker_already_gone pid=%s",
                getattr(proc, "pid", None),
            )
        try:
            proc.wait(timeout=_COSYVOICE_TERMINATE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            logger.warning(
                "event=cosyvoice_worker_unreaped pid=%s", getattr(proc, "pid", None)
            )
    except OSError:
        pass


def _cosyvoice_request(
    proc: subprocess.Popen[str], request: dict[str, Any]
) -> dict[str, Any]:
    if proc.poll() is not None:
        stderr_tail = getattr(proc, "_cosyvoice_stderr_tail", deque())
        message = _cosyvoice_worker_error(
            proc, stderr_tail, f"worker exited with code {proc.returncode}"
        )
        _get_cosyvoice_worker.cache_clear()
        raise RuntimeError(f"CosyVoice worker exited unexpectedly: {message}")
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    response_line = proc.stdout.readline()
    if not response_line:
        stderr_tail = getattr(proc, "_cosyvoice_stderr_tail", deque())
        message = _cosyvoice_worker_error(
            proc, stderr_tail, f"worker exited with code {proc.poll()}"
        )
        # Reap before dropping the cached handle so the next call starts a fresh
        # worker instead of competing with a hung one for GPU memory.
        _terminate_cosyvoice_worker(proc)
        _get_cosyvoice_worker.cache_clear()
        raise RuntimeError(f"CosyVoice worker closed its output: {message}")
    try:
        return json.loads(response_line)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "CosyVoice worker returned invalid response data: "
            f"{response_line.rstrip()!r}"
        ) from error


@lru_cache(maxsize=2)
def _get_chatterbox_turbo(device: str, model_dir: str | None, nano: bool):
    """Load Turbo/Nano lazily so EdgeTTS/Kokoro runs do not import its model stack."""
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    if model_dir:
        return ChatterboxTurboTTS.from_local(model_dir, device=device)
    return ChatterboxTurboTTS.from_pretrained(device=device, nano=nano)


async def synthesize_turn(
    turn: NormalizedTurn, out_dir: Path, config: dict[str, Any]
) -> Path:
    """Synthesize one turn with the engine selected in ``tts.engine``."""
    engine = get_engine(config)
    if engine == "edge_tts":
        import edge_tts  # Lazy import: only needed when Edge TTS is selected.

        voice = config["speaker_voice_map"][turn.speaker]
        output_path = out_dir / f"turn_{turn.turn_id:03d}.mp3"
        logger.debug(
            "event=turn_tts_start engine=edge_tts turn=%d speaker=%s voice=%s rate=%s",
            turn.turn_id,
            turn.speaker,
            voice,
            turn.rate,
        )
        await edge_tts.Communicate(
            text=turn.text,
            voice=voice,
            rate=rate_to_edge_tts(turn.rate),
        ).save(str(output_path))
        logger.debug(
            "event=turn_tts_complete engine=edge_tts turn=%d output=%s bytes=%d",
            turn.turn_id,
            output_path.name,
            output_path.stat().st_size,
        )
        return output_path

    output_path = out_dir / f"turn_{turn.turn_id:03d}.wav"
    if engine == "kokoro":
        kokoro = config["tts"]["kokoro"]
        voice = kokoro["voice_map"][turn.speaker]
        logger.debug(
            "event=turn_tts_start engine=kokoro turn=%d speaker=%s voice=%s rate=%s",
            turn.turn_id,
            turn.speaker,
            voice,
            turn.rate,
        )
        pipeline = _get_kokoro_pipeline(kokoro["lang_code"], kokoro.get("device"))
        chunks = [
            np.asarray(result.audio)
            for result in pipeline(
                turn.text,
                voice=voice,
                speed=rate_to_kokoro_speed(turn.rate),
            )
        ]
        if not chunks:
            raise RuntimeError(f"Kokoro did not generate audio for turn {turn.turn_id}")
        sf.write(output_path, np.concatenate(chunks), kokoro["sample_rate"])
        logger.debug(
            "event=turn_tts_complete engine=kokoro turn=%d output=%s chunks=%d bytes=%d",
            turn.turn_id,
            output_path.name,
            len(chunks),
            output_path.stat().st_size,
        )
        return output_path

    if engine == "cosyvoice":
        preflight_cosyvoice_controls(turn)
        cosy_cfg = config["tts"]["cosyvoice"]
        voice = cosy_cfg["voice_map"][turn.speaker]
        root = _project_root()
        repo_dir = _resolve_path(
            root, cosy_cfg.get("repo_dir", "third_party/CosyVoice")
        )
        model_dir = _resolve_path(
            root, cosy_cfg.get("model_dir", "models/Fun-CosyVoice3-0.5B")
        )
        prompt_wav = _resolve_path(root, voice["prompt_wav"])
        python_bin = _resolve_path(
            root, cosy_cfg.get("python_bin", "third_party/CosyVoice/.venv/bin/python")
        )
        required_paths = {
            "CosyVoice worker": _COSYVOICE_WORKER_SCRIPT,
            "CosyVoice Python interpreter": python_bin,
            "CosyVoice repository": repo_dir,
            "CosyVoice model directory": model_dir,
            "CosyVoice prompt audio": prompt_wav,
        }
        missing_paths = [
            f"{label}: {path}"
            for label, path in required_paths.items()
            if not path.exists()
        ]
        if missing_paths:
            raise RuntimeError(
                "CosyVoice cannot start because required paths are missing: "
                + "; ".join(missing_paths)
            )
        request = build_cosyvoice_request(
            turn,
            voice["prompt_text"],
            prompt_wav,
            output_path,
        )
        logger.debug(
            "event=turn_tts_start engine=cosyvoice turn=%d speaker=%s "
            "requested_rate=%s speed=%.3f mode=%s has_instruction=%s",
            turn.turn_id,
            turn.speaker,
            turn.rate,
            request["speed"],
            request["mode"],
            "instruction" in request,
        )
        worker = _get_cosyvoice_worker(
            str(python_bin),
            str(repo_dir),
            str(model_dir),
            cosy_cfg.get("load_trt", False),
            cosy_cfg.get("load_vllm", False),
            cosy_cfg.get("fp16", True),
        )
        response = _cosyvoice_request(worker, request)
        if response.get("status") != "ok":
            raise RuntimeError(
                f"CosyVoice failed for turn {turn.turn_id}: {response.get('message')}"
            )
        logger.debug(
            "event=turn_tts_complete engine=cosyvoice turn=%d output=%s bytes=%d",
            turn.turn_id,
            output_path.name,
            output_path.stat().st_size,
        )
        return output_path

    turbo = config["tts"]["chatterbox_turbo"]
    reference_audio = turbo.get("reference_audio_map", {}).get(turn.speaker)
    logger.debug(
        "event=turn_tts_start engine=chatterbox_turbo turn=%d speaker=%s reference_audio=%s",
        turn.turn_id,
        turn.speaker,
        reference_audio or "builtin",
    )
    model = _get_chatterbox_turbo(
        turbo["device"], turbo.get("model_dir"), turbo.get("variant", "turbo") == "nano"
    )
    kwargs = {
        "temperature": turbo["temperature"],
        "top_p": turbo["top_p"],
        "top_k": turbo["top_k"],
        "repetition_penalty": turbo["repetition_penalty"],
    }
    if reference_audio:
        kwargs["audio_prompt_path"] = reference_audio
    audio = model.generate(turn.text, **kwargs)
    sf.write(output_path, audio.squeeze(0).detach().cpu().numpy(), model.sr)
    logger.debug(
        "event=turn_tts_complete engine=chatterbox_turbo turn=%d output=%s bytes=%d",
        turn.turn_id,
        output_path.name,
        output_path.stat().st_size,
    )
    return output_path


async def synthesize_all_turns(
    turns: list[NormalizedTurn], out_dir: Path, config: dict[str, Any]
) -> dict[int, Path]:
    return {
        turn.turn_id: await synthesize_turn(turn, out_dir, config) for turn in turns
    }


def describe_engine(config: dict[str, Any]) -> dict[str, Any]:
    """Return an engine and voice configuration snapshot for metadata traceability."""
    engine = get_engine(config)
    if engine == "edge_tts":
        return {"engine": "edge_tts", "voices": config["speaker_voice_map"]}
    if engine == "kokoro":
        kokoro = config["tts"]["kokoro"]
        return {
            "engine": "kokoro",
            "model": "Kokoro-82M",
            "sample_rate": kokoro["sample_rate"],
            "voices": kokoro["voice_map"],
        }
    if engine == "cosyvoice":
        cosy_cfg = config["tts"]["cosyvoice"]
        configured_sample_rate = cosy_cfg.get("sample_rate")
        return {
            "engine": "cosyvoice",
            "model": "Fun-CosyVoice3-0.5B",
            "mode": "per_turn",
            "available_modes": ["zero_shot", "instruct2"],
            "control_mapping": "provisional",
            "model_dir": cosy_cfg.get("model_dir", "models/Fun-CosyVoice3-0.5B"),
            "repo_dir": cosy_cfg.get("repo_dir", "third_party/CosyVoice"),
            "fp16": cosy_cfg.get("fp16", True),
            "load_trt": cosy_cfg.get("load_trt", False),
            "load_vllm": cosy_cfg.get("load_vllm", False),
            # The worker reports its own sample rate over the protocol, but
            # synthesize_turn returns only a path, so the runtime value cannot be
            # recorded without changing that API. Until then the expected rate is
            # a declaration, never an observation of the written audio.
            "expected_sample_rate": (
                configured_sample_rate
                if configured_sample_rate is not None
                else _COSYVOICE3_EXPECTED_SAMPLE_RATE
            ),
            "expected_sample_rate_source": (
                "config" if configured_sample_rate is not None else "model_default"
            ),
            "runtime_sample_rate": None,
            "sample_rate_verification": _SAMPLE_RATE_NOT_RUNTIME_VERIFIED,
            "voices": {
                speaker: voice["prompt_wav"]
                for speaker, voice in cosy_cfg["voice_map"].items()
            },
            "prompts": {
                speaker: {
                    "prompt_wav": voice["prompt_wav"],
                    "prompt_text": voice["prompt_text"],
                }
                for speaker, voice in cosy_cfg["voice_map"].items()
            },
        }

    turbo = config["tts"]["chatterbox_turbo"]
    variant = turbo.get("variant", "turbo")
    return {
        "engine": "chatterbox_turbo",
        "model": f"chatterbox-{variant}",
        "device": turbo["device"],
        "reference_audio": turbo.get("reference_audio_map", {}),
        "generation_params": {
            "temperature": turbo["temperature"],
            "top_p": turbo["top_p"],
            "top_k": turbo["top_k"],
            "repetition_penalty": turbo["repetition_penalty"],
        },
    }
