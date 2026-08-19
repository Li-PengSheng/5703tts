# 5703 TTS Pipeline

`5703tts` renders crisis-dialogue JSON into aligned audio datasets. Each dialogue is validated, synthesized turn by turn, assembled without overlapping turns, exported as clean and telephone-quality WAV files, and accompanied by turn-level metadata and QC results.

The distribution and command are named `5703tts`. Its import package is `tts5703`, because Python package names cannot start with a digit.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- FFmpeg available on `PATH` (used by pydub to read/write compressed audio)

Install the locked project environment and run the command:

```bash
uv sync
uv run 5703tts
```

The first use of a model-backed engine may download model files. `edge_tts` requires network access for synthesis.

## Quick start

1. Put one or more dialogue JSON files in `data/input/`.
2. Choose and configure an engine in `config/config.yaml`.
3. Run the batch:

```bash
uv run 5703tts

# Override the default locations
uv run 5703tts --input path/to/input --output path/to/output

# Include per-turn TTS diagnostics in the console
uv run 5703tts --verbose
```

Detailed logs are written to `logs/run_YYYY-MM-DD.log`. Use `--config` or `--log-dir` to select alternate files or directories.

## Input format

Input files must conform to [`schemas/dialogue_schema.json`](schemas/dialogue_schema.json). A minimal dialogue is:

```json
{
  "dialogue_id": "example_001",
  "turns": [
    {
      "turn_id": 1,
      "speaker": "counsellor",
      "text": "Hello, how can I help?",
      "label": "normal"
    },
    {
      "turn_id": 2,
      "speaker": "caller",
      "text": "I need to talk to someone.",
      "label": "alert",
      "pause_after_ms": 700
    }
  ]
}
```

`turn_id` values must be unique and increasing. `speaker` must have a configured voice, and `label` must be one of `normal`, `alert`, or `confirm`. Optional turn fields are `rate` (`+/-N%`), `pause_after_ms`, `emotion`, `arousal`, and `paralinguistic_events`.

## TTS engines

Set `tts.engine` in `config/config.yaml` to one of the following:

| Engine | Turn format | Notes |
| --- | --- | --- |
| `edge_tts` | MP3 | Online Microsoft Edge TTS. Set voices in `speaker_voice_map`. |
| `kokoro` | WAV | Local/cached Kokoro model. Set voices in `tts.kokoro.voice_map`. |
| `chatterbox_turbo` | WAV | Chatterbox Turbo or Nano. Supports optional role-specific voice-reference clips. |

For example, to use Kokoro:

```yaml
tts:
  engine: kokoro
```

To use Chatterbox on CPU, change the engine and device:

```yaml
tts:
  engine: chatterbox_turbo
  chatterbox_turbo:
    device: cpu
```

Chatterbox uses its built-in voice unless `tts.chatterbox_turbo.reference_audio_map` supplies a reference clip for a speaker. For consistent, distinct role voices, provide a different reference clip longer than five seconds for each role. Set `model_dir` when loading a previously downloaded local checkpoint.

## Output

Each input dialogue writes to `data/output/<dialogue_id>/` by default:

- `turn_001.mp3` or `turn_001.wav` (one file per turn, depending on engine)
- `<dialogue_id>_clean.wav` (assembled speech and configured pauses)
- `<dialogue_id>_telephone.wav` (8 kHz mono, band-pass telephone treatment by default)
- `<dialogue_id>_metadata.json` (audio file names, TTS settings, labels, and timestamps)

The assembly uses direct joins plus short fades; turns are never crossfaded. Metadata timestamps therefore align with the non-overlapping turn boundaries. Generated audio and logs are intentionally ignored by Git.

## Configuration

[`config/config.yaml`](config/config.yaml) controls speaker-to-voice mappings, default rate and pauses, engine-specific settings, fades, and telephone filtering. The checked-in defaults use Kokoro at 24 kHz for synthesis and produce 8 kHz mono telephone audio.

## Tests

Run the fast, offline validation tests with:

```bash
uv run --with pytest pytest -q
```
