# Production architecture

## One input, one pipeline

`5703tts` accepts only the final nested JSON/JSONL contract supplied by
`upstream/Controlled-TTS-v1`.

```text
JSON / JSONL containers
  -> InputRecord
  -> validate_dialogue
  -> exclusion decision
  -> speaker sidecar lookup
  -> CanonicalDialogue / CanonicalTurn
  -> prepare selected backend plan
       -> HiggsPreparedTurn
       OR CosyVoicePreparedTurn
  -> TurnRenderResult
  -> common prepared assembly
  -> clean + telephone WAV
  -> final metadata + structural QC
  -> manifest v2 / semantic resume
```

Every parsed record is validated directly against this single upstream contract.

## Backend policy

- **Higgs:** primary production backend.
- **CosyVoice3:** explicit backup/secondary backend.
`tts.engine` selects Higgs or CosyVoice for the complete batch. A failure never causes
automatic backend switching. Config validation and rendering require only the selected
backend block.

## Canonical and backend boundaries

`CanonicalTurn` owns source semantics: ordinal, source turn ID, text, labels, upstream
and logical roles, scenario and render speaker IDs, requested controls, normalized
rate/arousal/affect, pauses, and hesitations. It contains no model input, native model
token, instruction, or speed argument.

Backend prepared turns cache the exact execution plan. Higgs continues to use the
frozen `map_turn_to_higgs()` implementation. CosyVoice caches text, prompt identity,
speed, mode, instruction, capability state, and realization status. Execution,
metadata, and QC consume cached plans and do not remap controls.

CosyVoice `pause_within > 0` is explicitly unsupported and fails preflight before an
output directory is created. Pause-before remains assembly-owned `0/500/900 ms`, and
turn WAV files contain speech only.

## Speaker contract

Final source records are immutable. The production speaker sidecar maps each logical
role to a production `spk_*` identity and target-specific references:

```json
{
  "caller": {
    "upstream_role": "User",
    "upstream_scenario_speaker_id": "C001",
    "render_speaker_id": "spk_001",
    "higgs_reference": {
      "reference_wav": "refs/higgs.wav",
      "sha256": "..."
    },
    "cosyvoice_reference": {
      "prompt_wav": "refs/cosy.wav",
      "prompt_text": "exact transcript",
      "sha256": "..."
    }
  }
}
```

The materializer target is explicit: `higgs`, `cosyvoice`, or `both`. Higgs references
must be separately approved; CosyVoice uses the registry primary reference. Neither is
substituted for the other.

## Batch, exclusion, and resume

Malformed JSONL rows become `input_error` entries while later valid rows continue.
Duplicate parsed dialogue IDs and unsafe output IDs fail before rendering. Exclusion is
data-driven and happens before speaker/preparation requirements.

Manifest v2 is the only production manifest. Per-dialogue fingerprints use canonical
record content, shared output settings, selected backend semantic/runtime identity, and
selected role/reference identity. Whole-file config formatting, input line location,
and unselected backend state are not semantic inputs. Before resume, the selected
reference WAVs are resolved with production path rules and their live SHA-256 values
are checked.

## Evidence boundary

Offline tests verify structure, control integrity, cached requests, artifact ordering,
timing, metadata, QC, and resume behavior. They do not prove emotion, arousal, speaker
similarity, intelligibility, or naturalness. Real Higgs GPU/cloud validation is the
next runtime gate.
