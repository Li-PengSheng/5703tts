# 5703tts

Production dialogue rendering for the Controlled-TTS-v1 upstream contract.

The `5703tts` command has one production input and one production pipeline:

```text
final JSON/JSONL
  -> InputRecord + final validation
  -> exclusion policy + speaker sidecar
  -> CanonicalDialogue
  -> selected cached backend plan
  -> common assembly, metadata, QC, manifest, and resume
```

Higgs is the primary production backend. CosyVoice3 is the supported
backup/secondary backend. `tts.engine` selects one backend for the complete run;
there is no automatic fallback.

Real Higgs reference-conditioned GPU validation remains the next runtime gate.
Offline tests do not establish naturalness, intelligibility, emotion fidelity, or
speaker similarity.

## Production input

The canonical handoff from `upstream/Controlled-TTS-v1` is final nested JSON or
JSONL. A `.json` file contains one dialogue; JSONL contains one dialogue object per
nonblank line. Directories may contain both supported container types.

Each turn keeps its upstream `speaker` (`User` or `Listener`) and nested controls:

```json
{
  "schema_version": "1.0",
  "dialogue_id": "corpus_v1_000001",
  "scenario": {
    "speakers": {
      "caller": {"speaker_id": "C001"},
      "counsellor": {"speaker_id": "L001"}
    }
  },
  "turns": [
    {
      "turn_id": "turn-a",
      "speaker": "User",
      "text": "I need help.",
      "labels": {},
      "acoustic": {
        "required": {
          "rate": "normal",
          "arousal": 2,
          "affect": "anxious",
          "pause_before": "short",
          "pause_within": 0,
          "hesitations": 0
        },
        "best_effort": {}
      }
    }
  ]
}
```

Legacy flat JSON and schema v0.2 are not accepted by the production CLI. Their
implementation remains temporarily in the repository only for U3 cleanup.

## Speaker materialization

Speaker assignment does not rewrite final source records. Materialize a backend-aware
sidecar with an explicit target:

```bash
uv run python scripts/materialize_speaker_assignments.py \
  --input data/final/dialogues.jsonl \
  --assignments data/speaker_assignments.jsonl \
  --registry data/speaker_pool/vctk_v0.1/speaker_registry.json \
  --active-speakers data/speaker_pool/vctk_v0.1/active_speakers.json \
  --output data/materialized-unused \
  --config-out config/materialized.yaml \
  --base-config config/config.yaml \
  --manifest data/speaker_sidecar.json \
  --backend both
```

Targets are independent:

- `--backend higgs` requires and emits only approved, hash-pinned
  `higgs_reference` values.
- `--backend cosyvoice` requires and emits only registry `primary_reference`
  values as `cosyvoice_reference`, preserving exact prompt text.
- `--backend both` requires and emits both reference types.

References are never substituted across backends. Production `spk_*` render IDs live
in the sidecar; upstream scenario speaker IDs and source turn speakers remain intact.

## Configuration and rendering

[`config/config.yaml`](config/config.yaml) is the production-oriented dual-backend
template. It selects Higgs and contains explicit external-asset placeholders. Provision
those paths before real execution. To select the backup backend for the whole run:

```yaml
tts:
  engine: cosyvoice
```

[`config/config_cosyvoice.yaml`](config/config_cosyvoice.yaml) is the explicit
backup-only example.

Render with the sidecar:

```bash
uv run 5703tts \
  --input data/final/dialogues.jsonl \
  --speaker-sidecar data/speaker_sidecar.json \
  --exclusion-policy data/exclusions.json \
  --config config/config.yaml \
  --output data/output \
  --resume
```

Only the selected backend's runtime block and references are required. Changing the
engine, selected runtime semantics, shared audio processing, prompt text, or selected
reference identity invalidates resume. Resume also hashes the selected reference's
live bytes before accepting previous output. Unselected backend configuration and
references do not interfere.

The production manifest is v2 and records rendered, resumed, excluded, input-error,
and render-failed outcomes. It is written atomically.

## Control boundary

Both backends consume the same canonical controls. Higgs retains the frozen
Controlled-TTS-v1 planner. CosyVoice3 uses `0.8/1.0/1.2` speeds and its provisional
instruction mapping. Pause-before uses shared `0/500/900 ms` assembly timing and
hesitations use the shared lexical planner.

CosyVoice3 has no evidence-backed deterministic realization for `pause_within > 0`;
such a dialogue fails closed before output-directory creation. The system does not
claim perceptual equivalence or validated acoustic fidelity.

## Development checks

```bash
uv sync
uv run --with pytest pytest -q
uv run ruff check .
git diff --check
```

No model, worker, GPU, network, or download is used by the test suite.

See [Architecture](docs/ARCHITECTURE.md),
[Chinese walkthrough](docs/PROJECT_WALKTHROUGH_CN.md),
[code reading guide](docs/CODE_READING_GUIDE.md), and
[Higgs cloud validation](docs/HIGGS_CLOUD_VALIDATION.md).
