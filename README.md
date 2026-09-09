# 5703 TTS Pipeline

`5703tts` renders crisis-dialogue JSON into aligned audio datasets. Each dialogue is validated, synthesized turn by turn, assembled without overlapping turns, exported as clean and telephone-quality WAV files, and accompanied by turn-level metadata and QC results.

The distribution and command are named `5703tts`. Its import package is `tts5703`, because Python package names cannot start with a digit.

## Project layout

| Path | Purpose | Git policy |
| --- | --- | --- |
| `src/tts5703/` | Validation, TTS orchestration, assembly, post-processing, metadata, QC, and CLI | Tracked |
| `schemas/` | Canonical dialogue input schema | Tracked |
| `config/` | Engine, voice, pause, and telephone-processing configuration | Tracked |
| `data/input/` | Dialogue JSON inputs | Tracked |
| `data/output/` | Generated turn audio, assembled audio, metadata, and QC output | Generated files ignored |
| `tests/` | Offline validation and worker-boundary tests | Tracked |
| `third_party/` | External TTS source checkouts and their private environments | Entire directory ignored |
| `models/` | Downloaded model snapshots and weights | Entire directory ignored |

`third_party/` and `models/` are intentionally not Git submodules and are not included when this repository is cloned. They currently occupy roughly 12 GB and 9.1 GB respectively in the working installation, so allow at least 25 GB of free space for setup, caches, and generated output.

## Requirements

- Git
- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- FFmpeg available on `PATH` (used by pydub to read/write compressed audio)

The configured CosyVoice engine additionally needs:

- Python 3.10 for its isolated environment
- SoX and its development libraries on Linux (`sox` and `libsox-dev` on Debian/Ubuntu)
- An NVIDIA GPU and CUDA-compatible driver for practical synthesis with the configured `fp16: true`; the pinned CosyVoice requirements install PyTorch CUDA 12.1 builds on Linux
- Network access during initial source, dependency, and model downloads

Without CUDA, CosyVoice detects the missing device and falls back to FP32 CPU inference even when `fp16` is configured, but synthesis will be much slower and needs substantial RAM. The validation tests do not load a TTS model and do not require a GPU or network.

## Installation

### 1. Clone and install the project environment

```bash
git clone https://github.com/Li-PengSheng/5703tts.git
cd 5703tts
uv sync
```

This creates the main Python 3.11 environment from `pyproject.toml` and `uv.lock`. Do not install CosyVoice into this environment: the pipeline launches it as an isolated subprocess because its pinned dependencies differ from the main project.

### 2. Download the CosyVoice source

The current integration was inspected against CosyVoice commit `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`. Clone it recursively at the path expected by `config/config.yaml`:

```bash
git clone --recursive https://github.com/QwenAudio/CosyVoice.git third_party/CosyVoice
git -C third_party/CosyVoice checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
git -C third_party/CosyVoice submodule update --init --recursive
```

The recursive checkout is required because the worker imports `third_party/CosyVoice/third_party/Matcha-TTS`.

### 3. Create the isolated CosyVoice environment

```bash
uv python install 3.10
uv venv --python 3.10 third_party/CosyVoice/.venv
uv pip install \
  --python third_party/CosyVoice/.venv/bin/python \
  -r third_party/CosyVoice/requirements.txt
```

On Debian/Ubuntu, install the system audio tools if they are not already present:

```bash
sudo apt-get update
sudo apt-get install ffmpeg sox libsox-dev
```

The checked-in configuration uses the POSIX interpreter path `third_party/CosyVoice/.venv/bin/python`. On Windows, create the environment normally and change `tts.cosyvoice.python_bin` to the resulting `Scripts/python.exe` path.

### 4. Download the CosyVoice3 model

Download the [`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) snapshot into the shorter local directory name expected by the project:

```bash
mkdir -p models
third_party/CosyVoice/.venv/bin/python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='models/Fun-CosyVoice3-0.5B')"
```

Do not change the download target unless you also update `tts.cosyvoice.model_dir`. A complete snapshot contains `cosyvoice3.yaml` plus the `.pt`, `.onnx`, tokenizer, and auxiliary model files required at startup. The optional `CosyVoice-ttsfrd` package is not required by this pipeline; CosyVoice falls back to WeTextProcessing.

### 5. Check the local runtime assets

These checks should all succeed before selecting CosyVoice:

```bash
test -x third_party/CosyVoice/.venv/bin/python
test -d third_party/CosyVoice/third_party/Matcha-TTS
test -f third_party/CosyVoice/asset/zero_shot_prompt.wav
test -f models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml
```

The sample configuration points both roles at CosyVoice's bundled `zero_shot_prompt.wav`, so it is enough for a technical smoke test but both roles will sound the same. Before producing a real two-speaker corpus, supply two distinct, appropriately licensed reference clips and set each `prompt_text` to the exact transcript of its corresponding clip.

## Quick start

1. Put one or more dialogue JSON files in `data/input/`.
2. Confirm the desired `tts.engine` and its voice paths in `config/config.yaml`.
3. Run the batch:

```bash
uv run 5703tts

# Override the default locations
uv run 5703tts --input path/to/input --output path/to/output

# Include per-turn TTS diagnostics in the console
uv run 5703tts --verbose

# Retry failed/incomplete dialogues and skip verified completed ones
uv run 5703tts --resume
```

Detailed logs are written to `logs/run_YYYY-MM-DD.log`. Use `--config` or `--log-dir` to select alternate files or directories.

Every invocation that reaches dialogue discovery atomically writes
`<output>/batch_result.json`. This is the authoritative batch result: `success`
means every discovered dialogue succeeded, `partial_failure` means some succeeded
and some failed, and `failure` means none succeeded. The CLI exits `0` only for
`success`; either failure status exits `1`. A failed dialogue does not stop later
dialogues or remove successful outputs. Configuration and argument errors occur
before this batch boundary, exit non-zero, and may not produce a manifest.
Duplicate `dialogue_id` values in the current input set are also a pre-execution
failure: they are rejected after discovery and canonical validation, before any
dialogue is rendered or `--resume` skip/retry decisions run, because outputs use
`<output>/<dialogue_id>/`. The error lists every conflicting path. This does not
forbid reusing an ID in a later invocation.

Without `--resume`, every discovered dialogue is rendered again. Existing output
is not skipped.

`--resume` is opt-in and works at dialogue granularity on a later invocation. It
does not retry a failed dialogue repeatedly inside one process, and it does not
resume from a mid-dialogue turn.

When `--resume` is set:

- verified previous successes are skipped and still count as successful final state
- previous failures and incomplete/stale outputs are rendered again
- newly discovered inputs are rendered
- changed input bytes or a changed config file cause a rerender rather than reuse
- unresolved failure still exits `1`; all current dialogues successful exits `0`

A previous success is skipped only when the prior manifest records that success,
the current input SHA-256 and config SHA-256 match, backend/model/mapping identity
matches when recorded, and production QC still finds complete metadata, clean WAV,
telephone WAV, and referenced turn audio. The existence of an output directory or
a single WAV file is not enough.

If `<output>/batch_result.json` is absent, `--resume` renders every current
dialogue and writes a new manifest. If that file exists but is unreadable or
malformed, the command fails before synthesis, leaves the file in place, and tells
you to repair or remove it or to run without `--resume`.

## Speaker assignment and materialization

Deterministic speaker assignment is preprocessing. It is not part of `5703tts`.

```bash
# 1. Canonical v0.2 dialogues (role names in speaker)
# 2. Assign persistent speaker IDs
uv run python scripts/assign_dialogue_speakers.py \
  --input path/to/canonical \
  --output path/to/speaker_assignments.jsonl \
  --summary path/to/speaker_assignment_summary.json \
  --seed 5703

# 3. Resolve those IDs into render JSON + CosyVoice voice_map
uv run python scripts/materialize_speaker_assignments.py \
  --input path/to/canonical \
  --assignments path/to/speaker_assignments.jsonl \
  --output path/to/render_input \
  --config-out path/to/config.yaml \
  --base-config config/config.yaml

# 4. Render. Recover later with the same command plus --resume
uv run 5703tts \
  --input path/to/render_input \
  --output path/to/output \
  --config path/to/config.yaml
```

Materialization rewrites each turn's `speaker` field from a role name to the
assigned `speaker_id` and keys CosyVoice `voice_map` by those IDs. The same role
name can therefore map to different speakers in different dialogues. The helper
does not run TTS. `--output` must be a new or empty directory.

## Input format

Schema v0.2 is the preferred input format and is defined in [`schemas/dialogue_schema.json`](schemas/dialogue_schema.json). Acoustic controls describe model-independent intent under `acoustic_spec`:

```json
{
  "schema_version": "0.2",
  "dialogue_id": "example_001",
  "turns": [
    {
      "turn_id": 1,
      "speaker": "counsellor",
      "text": "Hello, how can I help?",
      "label": "normal",
      "acoustic_spec": {
        "rate": "normal",
        "pause_before_ms": 0,
        "pause_after_ms": 500,
        "arousal": "medium",
        "coarse_affect": null,
        "emotion": null,
        "paralinguistic_events": []
      }
    }
  ]
}
```

`turn_id` values must be unique and increasing. `speaker` must have a configured voice, and `label` must be one of `normal`, `alert`, or `confirm`. Versionless legacy input with flat acoustic fields and percentage rates is temporarily accepted. Mapping semantic controls such as `slow`, `normal`, and `fast` to TTS-specific parameters is intentionally deferred to backend adapters.

## TTS engines

The current configuration uses local CosyVoice3:

```yaml
tts:
  engine: cosyvoice
```

The pipeline has four engine code paths, but their runtime availability differs:

| Engine | Turn format | Additional setup and limitations |
| --- | --- | --- |
| `cosyvoice` | WAV | Current configured engine. Requires the ignored source checkout, isolated environment, downloaded model, and reference audio described above. |
| `edge_tts` | MP3 | Installed in the main environment. Requires network access for every synthesis run and voices in `speaker_voice_map`. |
| `kokoro` | WAV | Installed in the main environment. Downloads its model on first use, then can run from the local Hugging Face cache; voices are configured in `tts.kokoro.voice_map`. |
| `chatterbox_turbo` | WAV | Code remains for experimentation, but its dependencies are not installed or supported by the locked project environment. |

To use Kokoro instead, set:

```yaml
tts:
  engine: kokoro
```

CosyVoice runs in its own Python 3.10 environment configured by `tts.cosyvoice.python_bin`. `load_trt` and `load_vllm` are disabled in the sample configuration.

### CosyVoice3 production control mapping v1

The current `cosyvoice3_control_mapping` v1 implementation is frozen for production integration while its runtime fidelity status remains `provisional`. It keeps numeric rate speeds `0.8`/`1.0`/`1.2`, maps low/high arousal and the five primary coarse-affect labels through instructions, accepts medium arousal without an additional clause, and retains `distressed` only for legacy compatibility. See the [mapping decision and evidence note](docs/cosyvoice3_control_mapping_v1.md) for the exact evidence boundary, limitations, and change policy.

### Backend control capabilities and preflight

[`src/tts5703/engine_capabilities.py`](src/tts5703/engine_capabilities.py) is the single declaration of what each backend does with a requested acoustic control. Production metadata and the controlled benchmark both read it, so reported support cannot drift between them.

| Support | Meaning |
| --- | --- |
| `model_control` | Sent to the backend and expected to change acoustics |
| `provisional_model_control` | Sent through a mapping whose acoustic fidelity is not yet validated |
| `pipeline_timing` | Realised by this project's assembly stage, not by the backend |
| `unsupported` | Cannot be consumed by the backend; a requested value is ignored |

Canonical schema validation stays backend-independent: `coarse_affect` remains an open string, so corpus design is not limited by one backend's vocabulary. Backend-specific mappings are checked separately by a preflight step that runs before any audio is rendered. Under CosyVoice, a mapped value such as `coarse_affect: "anxious"` passes preflight, while an unmapped value fails with the supported mappings listed. Under Kokoro the same value is accepted, ignored, and reported as ignored, because Kokoro consumes no affect control at all.

## Output

Each input dialogue writes to `data/output/<dialogue_id>/` by default:

- `turn_001.mp3` or `turn_001.wav` (one file per turn, depending on engine)
- `<dialogue_id>_clean.wav` (assembled speech and configured pauses)
- `<dialogue_id>_telephone.wav` (8 kHz mono, band-pass telephone treatment by default)
- `<dialogue_id>_metadata.json` (audio file names, TTS settings, labels, and timestamps)

The output root also contains `batch_result.json`, which reports the status,
input path, output directory, and concise failure information for each discovered
dialogue. It complements rather than replaces the detailed per-dialogue metadata.
Resume runs add `resume_requested`, `config_sha256`, per-dialogue `input_sha256`,
and an `action` of `rendered`, `retried`, or `skipped_completed`. Skipped
completed dialogues remain `status: success`.

The assembly uses direct joins plus short fades; turns are never crossfaded. Metadata timestamps therefore align with the non-overlapping turn boundaries. Generated audio and logs are intentionally ignored by Git.

Metadata separates requested intent from backend behaviour. Every turn keeps its flat requested fields for compatibility, repeats them under `requested_acoustic_spec`, and lists `ignored_requested_controls` for requested controls the selected backend cannot consume. `tts.control_support` records the whole capability map for the engine used; both fields are `null` for engines with no capability declaration (EdgeTTS, Chatterbox Turbo) so that "nothing declared" cannot be read as "nothing ignored". Under CosyVoice, `tts` also records the reproducibility-relevant configuration: `model_dir`, `repo_dir`, `fp16`, `load_trt`, `load_vllm`, and each speaker's `prompt_wav` and `prompt_text`. Sample rate is reported as a declaration rather than an observation. `expected_sample_rate` is Fun-CosyVoice3-0.5B's official 24 kHz output rate, or the optional `tts.cosyvoice.sample_rate` override, with `expected_sample_rate_source` stating which of the two applies. The worker reports its own rate over the protocol, but `synthesize_turn` returns only a path, so `runtime_sample_rate` stays `null` and `sample_rate_verification` is `not_runtime_verified`: no field claims to describe the audio actually written. Kokoro is different, because the pipeline itself writes Kokoro audio at the configured `tts.kokoro.sample_rate`, so that value is reported directly.

## Configuration

[`config/config.yaml`](config/config.yaml) controls speaker-to-voice mappings, default rate and pauses, engine-specific settings, fades, and telephone filtering. The current configuration uses CosyVoice3 and produces 8 kHz mono telephone audio in addition to the clean output. The Kokoro block specifies its 24 kHz synthesis settings when it is selected.

CosyVoice paths are resolved relative to the repository root:

| Configuration key | Default local target |
| --- | --- |
| `tts.cosyvoice.python_bin` | `third_party/CosyVoice/.venv/bin/python` |
| `tts.cosyvoice.repo_dir` | `third_party/CosyVoice` |
| `tts.cosyvoice.model_dir` | `models/Fun-CosyVoice3-0.5B` |
| `tts.cosyvoice.voice_map.*.prompt_wav` | A local reference WAV, currently CosyVoice's bundled sample |

## Git and large local assets

The root `.gitignore` excludes `/models/` and `/third_party/` completely. This prevents model weights, external repositories, nested Git metadata, and the large CosyVoice virtual environment from being added to the project repository. The generic `*.pt` and `*.safetensors` rules remain as a second safeguard for weights placed elsewhere.

Consequently, a fresh clone is intentionally incomplete for CosyVoice synthesis until the installation steps above are run. Do not commit downloaded weights or vendor the CosyVoice checkout into this repository. Updating either external dependency should be a deliberate local operation followed by updating the documented revision and retesting the integration.

## Tests

Run the fast, offline validation tests with:

```bash
uv run --with pytest pytest -q
```
