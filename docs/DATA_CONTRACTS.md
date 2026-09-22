# Data contracts

This document separates persisted JSON contracts from internal Python representations. Examples omit repeated backend-plan detail but do not invent fields; `<...>` denotes a value supplied by the current run.

## A. Final source dialogue — external persisted contract

A `.json` container holds one object. A `.jsonl` container holds one object per nonblank line. Parsing rejects duplicate keys, NaN, Infinity, non-object records, and blank `dialogue_id`. The production validator accepts only the final nested Controlled-TTS contract; legacy shapes are not adapted.

```json
{
  "schema_version": "1.0",
  "dialogue_id": "dialogue_001",
  "scenario": {
    "speakers": {
      "caller": {"speaker_id": "C001"},
      "counsellor": {"speaker_id": "L001"}
    }
  },
  "turns": [
    {
      "turn_id": "t01",
      "speaker": "User",
      "text": "I need some help.",
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

Required controls are `rate`, `arousal`, `affect`, `pause_before`, `pause_within`, and `hesitations`. Best-effort fields recognized by the mapping contract are `affect_fine`, `volume`, `flattened_affect`, and `events`; their presence is a request, not a realization guarantee.

`InputRecord.record_sha256` hashes canonical source content. Container path, JSON formatting, and JSONL line number are provenance and do not affect this hash. A malformed JSONL row becomes an `InputRecordFailure`; later rows remain independently readable.

## B. Speaker sidecar — external persisted contract

The materializer writes a sidecar and never rewrites source dialogue records. Top-level provenance identifies assignment, registry, active-speaker inputs, target backend, speakers, and dialogue entries. A representative `both` dialogue entry is:

```json
{
  "materialization_mode": "speaker_sidecar",
  "source_records_rewritten": false,
  "assignment_file": "data/speaker_assignments.jsonl",
  "assignment_sha256": "<64 hex>",
  "registry_path": "data/speaker_registry.json",
  "registry_version": "<registry version>",
  "registry_sha256": "<64 hex>",
  "active_speakers_path": "data/active_speakers.json",
  "active_speakers_sha256": "<64 hex>",
  "project_root": "<materialization root>",
  "input_count": 1,
  "backend_target": "both",
  "speakers": {
    "spk_001": {
      "primary_reference": {
        "prompt_wav": "refs/cv3.wav",
        "sha256": "<64 hex>"
      },
      "higgs_reference": {
        "reference_wav": "refs/higgs.wav",
        "sha256": "<64 hex>"
      }
    }
  },
  "dialogues": [
    {
      "source": {
        "container_path": "data/final/dialogues.jsonl",
        "format": "jsonl",
        "line_number": 1,
        "record_sha256": "<64 hex>"
      },
      "dialogue_id": "dialogue_001",
      "roles": {
        "caller": {
          "upstream_role": "User",
          "upstream_scenario_speaker_id": "C001",
          "render_speaker_id": "spk_001",
          "higgs_reference": {
            "reference_wav": "refs/higgs.wav",
            "sha256": "<64 hex>"
          },
          "cosyvoice_reference": {
            "prompt_wav": "refs/cv3.wav",
            "prompt_text": "You are a helpful assistant.<|endofprompt|>Exact transcript.",
            "sha256": "<64 hex>"
          }
        },
        "counsellor": {
          "upstream_role": "Listener",
          "upstream_scenario_speaker_id": "L001",
          "render_speaker_id": "spk_002",
          "higgs_reference": {
            "reference_wav": "refs/higgs_2.wav",
            "sha256": "<64 hex>"
          },
          "cosyvoice_reference": {
            "prompt_wav": "refs/cv3_2.wav",
            "prompt_text": "You are a helpful assistant.<|endofprompt|>Exact transcript.",
            "sha256": "<64 hex>"
          }
        }
      }
    }
  ]
}
```

For `backend_target: "higgs"`, only `higgs_reference` is required/emitted; for `cosyvoice`, only `cosyvoice_reference`; for `both`, both. Higgs approval is independent of registry `primary_reference`.

The renderer preserves the declared relative path for provenance, resolves it against the project root for access, and verifies live bytes against `sha256` before production use. Sidecar source `record_sha256`, upstream role, and scenario speaker ID must match the source record.

Resume identity uses selected per-role `upstream_role`, `render_speaker_id`, and selected reference identity. Sidecar location, unselected references, and source provenance path do not affect the render fingerprint.

## C. Canonical internal representation — Python only

`CanonicalDialogue` and `CanonicalTurn` are internal dataclasses; they are not persisted schemas.

`CanonicalDialogue` contains:

- `dialogue_id`, source `record_sha256`;
- ordered tuple of `CanonicalTurn`;
- `assignment_sha256`, `registry_sha256`, `active_speakers_sha256`.

`CanonicalTurn` contains:

- `ordinal`, `source_turn_id`, `text`, `labels`;
- upstream role, logical role, upstream scenario speaker ID, render speaker ID;
- raw `required` and `best_effort` requests;
- normalized `rate`, `arousal`, `affect`, `pause_before`;
- numeric `pause_before_ms`, `pause_within_count`, `hesitation_count`.

Canonical objects own source/render semantics but no backend-native fields. They do not contain Higgs model input/tokens or CosyVoice prompt/instruction/speed.

## D. Prepared backend representation — Python only

`PreparedDialogue` contains dialogue/source/provenance identity, one selected `engine`, and ordered prepared turns. It is internal, although its plans are copied into persisted metadata.

`HiggsPreparedTurn` contains source/speaker identity, selected `reference_wav`, resolved runtime path, reference SHA, and a validated cached plan. Important derived properties are:

- `model_input` — exact cached Higgs input;
- `synthesis_call_count` — required to equal one;
- `native_pause_token_count`;
- `rate` and `rate_plan` — frozen FFmpeg atempo decision;
- `pause_before_ms` — assembly-owned timing;
- `approved_reference` — render speaker plus declared/resolved path and SHA.

The internal/property names `approved_reference` and
`approved_speaker_reference` mean the reference selected by the prepared render
path and verified against its declared path/SHA. They do not by themselves mean
that the speaker/reference completed the project's formal Higgs
production-reference approval process. Formal backend-specific approval remains
represented by the speaker registry/sidecar workflow and project evidence
status.

`CosyVoicePreparedTurn` contains source/speaker identity, selected prompt WAV/resolved path/text/SHA, and a validated cached plan. Its `worker_request(output_path)` exposes only cached text, prompt, speed, mode, optional instruction, and output path.

Constructors call `validate_higgs_plan()` or `validate_cosyvoice_plan()` before publishing an object. Plans are deep-copied on construction and returned through defensive copies. Downstream execution, metadata, and QC must not remap or mutate them.

## E. TurnRenderResult — Python only

```text
TurnRenderResult(
    ordinal: int,
    source_turn_id: int | str,
    output_path: Path,
    synthesis_status: str,
    rate_status: str,
)
```

It links one prepared turn to its speech-only WAV and execution statuses. It does not own requested controls, backend planning, assembly pause, or perceptual assessment.

## F. Dialogue metadata — external persisted contract

Metadata is written as `<dialogue_id>_metadata.json`. Representative shape:

```json
{
  "schema_family": "final_nested",
  "dialogue_id": "dialogue_001",
  "clean_audio": "dialogue_001_clean.wav",
  "telephone_audio": "dialogue_001_telephone.wav",
  "tts": {
    "engine": "higgs",
    "mode": "prepared_controlled_tts_v1",
    "backend_identity": {
      "backend": "higgs",
      "control_mapping": {},
      "references": {
        "spk_001": {"reference_wav": "refs/higgs.wav", "sha256": "<64 hex>"}
      }
    },
    "model_id": "<contract model ID>",
    "control_mapping": {"mapping_version": "controlled_tts_v1", "implementation_id": "controlled_tts_v1_prod_1"},
    "generation_profile": {},
    "runtime": {},
    "runtime_verification": "not_runtime_verified",
    "references": {
      "spk_001": {"reference_wav": "refs/higgs.wav", "sha256": "<64 hex>"}
    },
    "control_support": {"required": {}, "best_effort": {}}
  },
  "provenance": {
    "record_sha256": "<64 hex>",
    "assignment_sha256": "<64 hex>",
    "registry_sha256": "<64 hex>",
    "active_speakers_sha256": "<64 hex>",
    "mapping_version": "controlled_tts_v1",
    "release_status": "PROVISIONAL_PRODUCTION_MAPPING_WEEK6_FREEZE",
    "contract_sha256": "<64 hex>",
    "implementation_id": "controlled_tts_v1_prod_1"
  },
  "turns": [
    {
      "source_identity": {
        "ordinal": 1,
        "source_turn_id": "t01",
        "upstream_role": "User",
        "logical_role": "caller",
        "upstream_scenario_speaker_id": "C001",
        "render_speaker_id": "spk_001"
      },
      "labels": {},
      "requested": {"acoustic": {"required": {}, "best_effort": {}}},
      "planned": {"mapping_version": "controlled_tts_v1"},
      "approved_speaker_reference": {
        "render_speaker_id": "spk_001",
        "reference_wav": "refs/higgs.wav",
        "resolved_reference_wav": "<resolved path>",
        "sha256": "<64 hex>"
      },
      "execution": {
        "synthesis_status": "synthesized",
        "rate_status": "not_required",
        "turn_audio": "turn_001.wav"
      },
      "timing": {
        "ordinal": 1,
        "source_turn_id": "t01",
        "pause_before_ms": 500,
        "pause_after_ms": 0,
        "start_sec": 0.5,
        "end_sec": 2.4
      }
    }
  ]
}
```

CosyVoice metadata uses `mode: "prepared_final"`, its mapping name/version/status, `model`, and prompt reference fields (`prompt_wav`, resolved prompt path, `prompt_text`, SHA). The exact `planned` object is backend-specific and is copied from the prepared turn.

Both backend metadata paths currently write
`runtime_verification: "not_runtime_verified"`. This field is deliberately
unchanged in code: it says that one dialogue's metadata does not attest the
external runtime environment. It does not say that the project has never
executed Higgs3 or CosyVoice3 on a real GPU. Project-level runtime status belongs
in [CURRENT_STATUS.md](CURRENT_STATUS.md) and evidence records.

Resume artifact validation uses metadata dialogue/source identity, selected backend and implementation ID, turn order/IDs, labels, requested controls, timing, and expected turn/output filenames. It also decodes the live WAVs. Metadata-controlled arbitrary paths are never used for cleanup.

## G. Batch manifest v2 — external persisted contract

`<output_root>/batch_result.json` has `manifest_version: "2.0"` and is the authoritative batch state. Representative structure:

```json
{
  "manifest_version": "2.0",
  "status": "success",
  "started_at": "<ISO timestamp>",
  "finished_at": "<ISO timestamp>",
  "duration_seconds": 12.345,
  "backend": "higgs",
  "backend_identity": {},
  "backend_identity_sha256": "<64 hex>",
  "config_path": "config/config.yaml",
  "config_sha256": "<64 hex>",
  "input_root": "data/final/dialogues.jsonl",
  "output_root": "data/output",
  "resume_requested": true,
  "batch_provenance": {
    "controlled_tts": {},
    "exclusion_policy": {"schema_version": "1.0", "policy_sha256": "<64 hex>"},
    "speaker_sidecar": {
      "path": "data/speaker_sidecar.json",
      "assignment_sha256": "<64 hex>",
      "registry_sha256": "<64 hex>",
      "active_speakers_sha256": "<64 hex>"
    },
    "input_containers": ["data/final/dialogues.jsonl"]
  },
  "records_total": 1,
  "dialogues_total": 1,
  "dialogues_succeeded": 1,
  "dialogues_failed": 0,
  "dialogues_rendered": 1,
  "dialogues_resumed": 0,
  "dialogues_excluded": 0,
  "input_failures": 0,
  "dialogues_skipped": 0,
  "summary": {
    "total_valid_records": 1,
    "rendered": 1,
    "resumed": 0,
    "excluded": 0,
    "failed": 0,
    "input_failures": 0
  },
  "results": [
    {
      "dialogue_id": "dialogue_001",
      "source": {
        "container_path": "data/final/dialogues.jsonl",
        "format": "jsonl",
        "line_number": 1,
        "record_sha256": "<64 hex>"
      },
      "action": "rendered",
      "status": "success",
      "render_fingerprint": "<64 hex>",
      "output_dir": "<resolved output directory>",
      "error": null,
      "exclusion": null
    }
  ]
}
```

Actions are `rendered`, `resumed`, `excluded_known_issue`, `input_error`, or `render_failed`. Overall status is `success`, `partial_failure`, or `failure`.

The per-record render fingerprint hashes source identity, shared render-affecting config, selected static backend identity, selected speaker/reference materialization, and exclusion decision. Input location/formatting and unselected backend configuration do not affect it. A matching prior successful result is necessary but not sufficient: live metadata/WAV/reference integrity must also pass.

The manifest is atomically published at batch completion. It is not checkpointed after every dialogue.

## H. Exclusion policy — external persisted contract

```json
{
  "schema_version": "1.0",
  "exclusions": [
    {
      "dialogue_id": "dialogue_known_issue_001",
      "reason": "Documented source issue",
      "provenance": {"ticket": "<external reference>"}
    }
  ]
}
```

`schema_version` must equal `1.0`; exclusions require unique nonblank `dialogue_id` and `reason`; `provenance` is optional JSON. The policy receives a canonical `policy_sha256`. Excluded records are decided before sidecar/preparation, have no render fingerprint or output directory, and appear in manifest results with status `excluded` and action `excluded_known_issue`.
