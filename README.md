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

The default engine is `edge_tts`, which requires network access for synthesis. Kokoro downloads its model on first use and can run from its local cache afterwards.

## Quick start

1. Put one or more dialogue JSON files in `data/input/`.
2. Use the default Edge TTS configuration, or switch to Kokoro in `config/config.yaml`.
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

The checked-in configuration uses online Edge TTS:

```yaml
tts:
  engine: edge_tts
```

Two engines are installed and supported:

| Engine | Turn format | Notes |
| --- | --- | --- |
| `edge_tts` | MP3 | **Default.** Online Microsoft Edge TTS. Set voices in `speaker_voice_map`. |
| `kokoro` | WAV | Local/cached Kokoro model. Set voices in `tts.kokoro.voice_map`. |

To use Kokoro instead, set:

```yaml
tts:
  engine: kokoro
```

The repository retains Chatterbox Turbo support code for future experimentation, but Chatterbox dependencies are deliberately not included in the project environment. It is therefore not a supported configured engine at present; selecting it requires manually restoring its dependencies and supplying its configuration.

## Output

Each input dialogue writes to `data/output/<dialogue_id>/` by default:

- `turn_001.mp3` or `turn_001.wav` (one file per turn, depending on engine)
- `<dialogue_id>_clean.wav` (assembled speech and configured pauses)
- `<dialogue_id>_telephone.wav` (8 kHz mono, band-pass telephone treatment by default)
- `<dialogue_id>_metadata.json` (audio file names, TTS settings, labels, and timestamps)

The assembly uses direct joins plus short fades; turns are never crossfaded. Metadata timestamps therefore align with the non-overlapping turn boundaries. Generated audio and logs are intentionally ignored by Git.

## Configuration

[`config/config.yaml`](config/config.yaml) controls speaker-to-voice mappings, default rate and pauses, engine-specific settings, fades, and telephone filtering. The checked-in defaults use Edge TTS and produce 8 kHz mono telephone audio. The Kokoro block specifies its 24 kHz synthesis settings when it is selected.

## Tests

Run the fast, offline validation tests with:

```bash
uv run --with pytest pytest -q
```
