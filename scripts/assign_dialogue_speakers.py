"""Deterministically assign dialogue roles to the frozen active speaker pool.

This is rendering configuration only. It does not synthesise audio, rewrite
dialogue JSON, or write ``speaker_id`` into ``acoustic_spec``. Crisis labels
and acoustic conditions are recorded after assignment for descriptive QA;
they never participate in the assignment decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEAKER_POOL_DIR = PROJECT_ROOT / "data" / "speaker_pool" / "vctk_v0.1"
DEFAULT_ACTIVE_SPEAKERS = SPEAKER_POOL_DIR / "active_speakers.json"
DEFAULT_REGISTRY = SPEAKER_POOL_DIR / "speaker_registry.json"
DEFAULT_OUTPUT = SPEAKER_POOL_DIR / "speaker_assignments.jsonl"
DEFAULT_SUMMARY = SPEAKER_POOL_DIR / "speaker_assignment_summary.json"
DEFAULT_SEED = 5703
V0_1_POOL_VERSION = "vctk_v0.1"
V0_1_ACTIVE_SPEAKER_COUNT = 10
NULL_COUNT_KEY = "null"
UNSPECIFIED_COUNT_KEY = "unspecified"
KNOWN_LABEL_ORDER = ("normal", "alert", "confirm")
JSONL_SEPARATORS = (", ", ": ")


class SpeakerAssignmentError(ValueError):
    """Raised when the active pool or a dialogue cannot be assigned as specified."""


@dataclass(frozen=True)
class Exclusion:
    speaker_id: str
    reason: str


@dataclass(frozen=True)
class ActivePool:
    speaker_pool_version: str
    status: str
    selection_basis: str
    active_speaker_ids: tuple[str, ...]
    exclusions: tuple[Exclusion, ...]


@dataclass(frozen=True)
class TurnRecord:
    speaker: str
    label: str | None
    rate: Any
    arousal: Any
    coarse_affect: Any


@dataclass(frozen=True)
class DialogueRecord:
    dialogue_id: str
    source_path: Path
    roles: tuple[str, ...]
    turns: tuple[TurnRecord, ...]


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SpeakerAssignmentError(f"{label} not found: {path}") from error
    except json.JSONDecodeError as error:
        raise SpeakerAssignmentError(
            f"{label} is not valid JSON ({path}): {error}"
        ) from error
    if not isinstance(value, dict):
        raise SpeakerAssignmentError(f"{label} must contain a JSON object: {path}")
    return value


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpeakerAssignmentError(f"{label} must be a non-empty string")
    return value.strip()


def load_registry(path: Path) -> dict[str, Any]:
    """Load speaker_registry.json; only speaker IDs and pool version are required."""
    registry = _load_json_object(path, "Speaker registry")
    speakers = registry.get("speakers")
    if not isinstance(speakers, list) or not speakers:
        raise SpeakerAssignmentError(f"Speaker registry lists no speakers: {path}")
    ids: list[str] = []
    seen: set[str] = set()
    for position, entry in enumerate(speakers, start=1):
        if not isinstance(entry, dict):
            raise SpeakerAssignmentError(
                f"Speaker registry entry {position} must be a JSON object"
            )
        speaker_id = _non_empty_string(
            entry.get("speaker_id"), f"Speaker registry entry {position} speaker_id"
        )
        if speaker_id in seen:
            raise SpeakerAssignmentError(
                f"Duplicate speaker_id in registry: {speaker_id}"
            )
        seen.add(speaker_id)
        ids.append(speaker_id)
    registry["_speaker_ids"] = ids
    return registry


def registry_speaker_ids(registry: dict[str, Any]) -> list[str]:
    ids = registry.get("_speaker_ids")
    if isinstance(ids, list):
        return list(ids)
    return [entry["speaker_id"] for entry in registry["speakers"]]


def load_and_validate_active_pool(active_path: Path, registry_path: Path) -> ActivePool:
    """Validate the frozen active pool against the materialised registry."""
    registry = load_registry(registry_path)
    active = _load_json_object(active_path, "Active speaker pool")
    registry_ids = set(registry_speaker_ids(registry))
    registry_version = registry.get("speaker_pool_version")
    pool_version = _non_empty_string(
        active.get("speaker_pool_version"), "speaker_pool_version"
    )
    if pool_version != registry_version:
        raise SpeakerAssignmentError(
            f"Active pool version {pool_version!r} does not match speaker registry "
            f"version {registry_version!r}"
        )

    raw_ids = active.get("active_speaker_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise SpeakerAssignmentError("active_speaker_ids must be a non-empty list")
    active_ids: list[str] = []
    seen_active: set[str] = set()
    for position, value in enumerate(raw_ids, start=1):
        speaker_id = _non_empty_string(value, f"active_speaker_ids[{position}]")
        if speaker_id in seen_active:
            raise SpeakerAssignmentError(f"Duplicate active speaker ID: {speaker_id}")
        if speaker_id not in registry_ids:
            raise SpeakerAssignmentError(
                f"Active speaker {speaker_id} is not in the speaker registry"
            )
        seen_active.add(speaker_id)
        active_ids.append(speaker_id)

    if pool_version == V0_1_POOL_VERSION and len(active_ids) != (
        V0_1_ACTIVE_SPEAKER_COUNT
    ):
        raise SpeakerAssignmentError(
            f"{V0_1_POOL_VERSION} requires exactly {V0_1_ACTIVE_SPEAKER_COUNT} "
            f"active speakers, found {len(active_ids)}"
        )

    raw_excluded = active.get("excluded_from_active_pool", [])
    if not isinstance(raw_excluded, list):
        raise SpeakerAssignmentError("excluded_from_active_pool must be a list")
    exclusions: list[Exclusion] = []
    seen_excluded: set[str] = set()
    for position, entry in enumerate(raw_excluded, start=1):
        if not isinstance(entry, dict):
            raise SpeakerAssignmentError(
                f"excluded_from_active_pool[{position}] must be a JSON object"
            )
        speaker_id = _non_empty_string(
            entry.get("speaker_id"),
            f"excluded_from_active_pool[{position}].speaker_id",
        )
        reason = _non_empty_string(
            entry.get("reason"), f"excluded_from_active_pool[{position}].reason"
        )
        if speaker_id in seen_excluded:
            raise SpeakerAssignmentError(f"Duplicate excluded speaker ID: {speaker_id}")
        if speaker_id not in registry_ids:
            raise SpeakerAssignmentError(
                f"Excluded speaker {speaker_id} is not in the speaker registry"
            )
        if speaker_id in seen_active:
            raise SpeakerAssignmentError(
                f"Speaker {speaker_id} cannot be both active and excluded"
            )
        seen_excluded.add(speaker_id)
        exclusions.append(Exclusion(speaker_id=speaker_id, reason=reason))

    status = active.get("status")
    selection_basis = active.get("selection_basis")
    return ActivePool(
        speaker_pool_version=pool_version,
        status=status.strip() if isinstance(status, str) else "",
        selection_basis=(
            selection_basis.strip() if isinstance(selection_basis, str) else ""
        ),
        active_speaker_ids=tuple(active_ids),
        exclusions=tuple(exclusions),
    )


def _count_key(value: Any) -> str:
    if value is None:
        return NULL_COUNT_KEY
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else UNSPECIFIED_COUNT_KEY
    return str(value)


def _acoustic_from_turn(turn: dict[str, Any]) -> tuple[Any, Any, Any]:
    spec = turn.get("acoustic_spec")
    if isinstance(spec, dict):
        return spec.get("rate"), spec.get("arousal"), spec.get("coarse_affect")
    return turn.get("rate"), turn.get("arousal"), turn.get("coarse_affect")


def load_dialogue(path: Path) -> DialogueRecord:
    """Read one dialogue JSON file without rewriting it or consulting TTS config."""
    raw = _load_json_object(path, "Dialogue")
    dialogue_id = _non_empty_string(raw.get("dialogue_id"), f"{path}: dialogue_id")
    turns_raw = raw.get("turns")
    if not isinstance(turns_raw, list) or not turns_raw:
        raise SpeakerAssignmentError(f"{path}: dialogue {dialogue_id} has no turns")
    turns: list[TurnRecord] = []
    roles: list[str] = []
    seen_roles: set[str] = set()
    for position, turn in enumerate(turns_raw, start=1):
        if not isinstance(turn, dict):
            raise SpeakerAssignmentError(
                f"{path}: {dialogue_id} turn {position} must be a JSON object"
            )
        speaker = _non_empty_string(
            turn.get("speaker"), f"{path}: {dialogue_id} turn {position} speaker"
        )
        if speaker not in seen_roles:
            seen_roles.add(speaker)
            roles.append(speaker)
        rate, arousal, coarse_affect = _acoustic_from_turn(turn)
        label = turn.get("label")
        turns.append(
            TurnRecord(
                speaker=speaker,
                label=label if isinstance(label, str) else None,
                rate=rate,
                arousal=arousal,
                coarse_affect=coarse_affect,
            )
        )
    return DialogueRecord(
        dialogue_id=dialogue_id,
        source_path=path,
        roles=tuple(roles),
        turns=tuple(turns),
    )


def discover_dialogue_paths(input_path: Path) -> list[Path]:
    """Discover dialogue JSON files with a path sort that ignores readdir order."""
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise SpeakerAssignmentError(f"Input path does not exist: {input_path}")
    paths = sorted(
        (path for path in input_path.rglob("*.json") if path.is_file()),
        key=lambda path: path.as_posix(),
    )
    if not paths:
        raise SpeakerAssignmentError(f"No JSON files found under {input_path}")
    return paths


def load_dialogues(input_path: Path) -> list[DialogueRecord]:
    """Load dialogues and sort them by dialogue_id, then by stable path."""
    dialogues = [load_dialogue(path) for path in discover_dialogue_paths(input_path)]
    seen: dict[str, Path] = {}
    for dialogue in dialogues:
        previous = seen.get(dialogue.dialogue_id)
        if previous is not None:
            raise SpeakerAssignmentError(
                f"Duplicate dialogue_id {dialogue.dialogue_id!r} in "
                f"{previous} and {dialogue.source_path}"
            )
        seen[dialogue.dialogue_id] = dialogue.source_path
    dialogues.sort(key=lambda item: (item.dialogue_id, item.source_path.as_posix()))
    return dialogues


def _tie_break_digest(seed: int, dialogue_id: str, role: str, speaker_id: str) -> bytes:
    """Stable SHA256 digest; never Python's process-salted built-in hash."""
    material = f"{seed}\0{dialogue_id}\0{role}\0{speaker_id}".encode()
    return hashlib.sha256(material).digest()


def choose_speaker_for_role(
    *,
    role: str,
    dialogue_id: str,
    seed: int,
    used_in_dialogue: set[str],
    usage_count: dict[str, int],
    role_usage_count: dict[str, dict[str, int]],
    active_speaker_ids: tuple[str, ...],
) -> str:
    """Pick one unused active speaker using usage, then role usage, then SHA256.

    Labels, arousal, affect, emotion, and rate are intentionally absent from
    this decision.
    """
    available = [
        speaker_id
        for speaker_id in active_speaker_ids
        if speaker_id not in used_in_dialogue
    ]
    if not available:
        raise SpeakerAssignmentError(
            f"{dialogue_id}: no remaining active speakers for role {role!r}"
        )
    min_usage = min(usage_count[speaker_id] for speaker_id in available)
    candidates = [
        speaker_id for speaker_id in available if usage_count[speaker_id] == min_usage
    ]
    role_counts = role_usage_count[role]
    min_role_usage = min(role_counts[speaker_id] for speaker_id in candidates)
    candidates = [
        speaker_id
        for speaker_id in candidates
        if role_counts[speaker_id] == min_role_usage
    ]
    return min(
        candidates,
        key=lambda speaker_id: (
            _tie_break_digest(seed, dialogue_id, role, speaker_id),
            speaker_id,
        ),
    )


def assign_dialogues(
    dialogues: list[DialogueRecord], pool: ActivePool, seed: int
) -> list[dict[str, Any]]:
    """Assign a distinct active speaker to each role in each dialogue."""
    active = pool.active_speaker_ids
    usage_count = {speaker_id: 0 for speaker_id in active}
    role_usage_count: dict[str, dict[str, int]] = defaultdict(
        lambda: {speaker_id: 0 for speaker_id in active}
    )
    assignments: list[dict[str, Any]] = []
    for dialogue in dialogues:
        if not dialogue.roles:
            raise SpeakerAssignmentError(
                f"{dialogue.dialogue_id}: no speaker roles found"
            )
        if len(dialogue.roles) > len(active):
            raise SpeakerAssignmentError(
                f"{dialogue.dialogue_id}: {len(dialogue.roles)} distinct roles "
                f"{list(dialogue.roles)} exceed the {len(active)} active speakers"
            )
        used: set[str] = set()
        role_assignments: dict[str, str] = {}
        for role in dialogue.roles:
            speaker_id = choose_speaker_for_role(
                role=role,
                dialogue_id=dialogue.dialogue_id,
                seed=seed,
                used_in_dialogue=used,
                usage_count=usage_count,
                role_usage_count=role_usage_count,
                active_speaker_ids=active,
            )
            if speaker_id not in active:
                raise SpeakerAssignmentError(
                    f"{dialogue.dialogue_id}: assigned inactive speaker {speaker_id}"
                )
            role_assignments[role] = speaker_id
            used.add(speaker_id)
            usage_count[speaker_id] += 1
            role_usage_count[role][speaker_id] += 1
        assignments.append(
            {
                "dialogue_id": dialogue.dialogue_id,
                "speaker_pool_version": pool.speaker_pool_version,
                "assignment_seed": seed,
                "role_assignments": role_assignments,
            }
        )
    return assignments


def _ordered_count_keys(keys: set[str], preferred: tuple[str, ...] = ()) -> list[str]:
    known = [key for key in preferred if key in keys]
    rest = sorted(key for key in keys if key not in preferred)
    return known + rest


def _zero_counts(keys: list[str]) -> dict[str, int]:
    return {key: 0 for key in keys}


def build_assignment_summary(
    *,
    dialogues: list[DialogueRecord],
    assignments: list[dict[str, Any]],
    pool: ActivePool,
    seed: int,
) -> dict[str, Any]:
    """Describe usage and label/acoustic exposure. Never feeds back into assignment."""
    active = pool.active_speaker_ids
    speaker_usage = {speaker_id: 0 for speaker_id in active}
    dialogues_per_speaker = {speaker_id: 0 for speaker_id in active}
    role_names: set[str] = set()
    for record in assignments:
        mapping: dict[str, str] = record["role_assignments"]
        seen_in_dialogue = set(mapping.values())
        for role, speaker_id in mapping.items():
            role_names.add(role)
            speaker_usage[speaker_id] += 1
        for speaker_id in seen_in_dialogue:
            dialogues_per_speaker[speaker_id] += 1

    roles_sorted = sorted(role_names)
    role_usage = {
        role: {speaker_id: 0 for speaker_id in active} for role in roles_sorted
    }
    assignment_by_id = {record["dialogue_id"]: record for record in assignments}
    label_keys: set[str] = set()
    rate_keys: set[str] = set()
    arousal_keys: set[str] = set()
    affect_keys: set[str] = set()
    per_speaker_labels: dict[str, dict[str, int]] = {spk: {} for spk in active}
    per_speaker_rate: dict[str, dict[str, int]] = {spk: {} for spk in active}
    per_speaker_arousal: dict[str, dict[str, int]] = {spk: {} for spk in active}
    per_speaker_affect: dict[str, dict[str, int]] = {spk: {} for spk in active}

    def bump(store: dict[str, dict[str, int]], speaker_id: str, key: str) -> None:
        store[speaker_id][key] = store[speaker_id].get(key, 0) + 1

    for dialogue in dialogues:
        mapping = assignment_by_id[dialogue.dialogue_id]["role_assignments"]
        for role, speaker_id in mapping.items():
            role_usage[role][speaker_id] += 1
        for turn in dialogue.turns:
            speaker_id = mapping[turn.speaker]
            label_key = _count_key(turn.label)
            rate_key = _count_key(turn.rate)
            arousal_key = _count_key(turn.arousal)
            affect_key = _count_key(turn.coarse_affect)
            label_keys.add(label_key)
            rate_keys.add(rate_key)
            arousal_keys.add(arousal_key)
            affect_keys.add(affect_key)
            bump(per_speaker_labels, speaker_id, label_key)
            bump(per_speaker_rate, speaker_id, rate_key)
            bump(per_speaker_arousal, speaker_id, arousal_key)
            bump(per_speaker_affect, speaker_id, affect_key)

    labels_sorted = _ordered_count_keys(label_keys, KNOWN_LABEL_ORDER)
    rates_sorted = _ordered_count_keys(rate_keys)
    arousals_sorted = _ordered_count_keys(
        arousal_keys, (NULL_COUNT_KEY, "low", "medium", "high")
    )
    affects_sorted = _ordered_count_keys(affect_keys, (NULL_COUNT_KEY,))

    label_exposure = {}
    acoustic_exposure = {}
    for speaker_id in active:
        labels = _zero_counts(labels_sorted)
        labels.update(per_speaker_labels[speaker_id])
        rates = _zero_counts(rates_sorted)
        rates.update(per_speaker_rate[speaker_id])
        arousals = _zero_counts(arousals_sorted)
        arousals.update(per_speaker_arousal[speaker_id])
        affects = _zero_counts(affects_sorted)
        affects.update(per_speaker_affect[speaker_id])
        label_exposure[speaker_id] = labels
        acoustic_exposure[speaker_id] = {
            "rate": rates,
            "arousal": arousals,
            "coarse_affect": affects,
        }

    return {
        "speaker_pool_version": pool.speaker_pool_version,
        "status": "descriptive_qa_only",
        "assignment_seed": seed,
        "input_dialogue_count": len(dialogues),
        "active_speaker_count": len(active),
        "active_speaker_ids": list(active),
        "excluded_from_active_pool": [
            {"speaker_id": item.speaker_id, "reason": item.reason}
            for item in pool.exclusions
        ],
        "speaker_usage": speaker_usage,
        "dialogues_per_speaker": dialogues_per_speaker,
        "role_usage": role_usage,
        "label_exposure": label_exposure,
        "acoustic_exposure": acoustic_exposure,
        "notes": [
            "Descriptive QA only. Assignments were not adjusted from these counts.",
            "Do not treat a small sample as evidence of statistical balance.",
            "Speaker assignment does not use labels or acoustic conditions as input.",
        ],
    }


def encode_assignment_jsonl(assignments: list[dict[str, Any]]) -> str:
    lines = [
        json.dumps(record, ensure_ascii=False, separators=JSONL_SEPARATORS)
        for record in assignments
    ]
    return "\n".join(lines) + "\n"


def encode_summary_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, ensure_ascii=False) + "\n"


def write_assignment_outputs(
    assignments: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
    summary_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encode_assignment_jsonl(assignments), encoding="utf-8")
    summary_path.write_text(encode_summary_json(summary), encoding="utf-8")


def run_assignment(
    *,
    input_path: Path,
    active_speakers_path: Path = DEFAULT_ACTIVE_SPEAKERS,
    registry_path: Path = DEFAULT_REGISTRY,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the pool, assign every discovered dialogue, and write manifests."""
    pool = load_and_validate_active_pool(active_speakers_path, registry_path)
    dialogues = load_dialogues(input_path)
    assignments = assign_dialogues(dialogues, pool, seed)
    summary = build_assignment_summary(
        dialogues=dialogues, assignments=assignments, pool=pool, seed=seed
    )
    write_assignment_outputs(assignments, summary, output_path, summary_path)
    return assignments, summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assign dialogue roles to frozen active speaker IDs without running TTS"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Dialogue JSON file or directory of dialogue JSON files",
    )
    parser.add_argument("--active-speakers", type=Path, default=DEFAULT_ACTIVE_SPEAKERS)
    parser.add_argument("--speaker-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        assignments, summary = run_assignment(
            input_path=args.input,
            active_speakers_path=args.active_speakers,
            registry_path=args.speaker_registry,
            output_path=args.output,
            summary_path=args.summary,
            seed=args.seed,
        )
    except (SpeakerAssignmentError, OSError) as error:
        print(
            f"Speaker assignment failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2

    print(f"Dialogues: {summary['input_dialogue_count']}")
    print(f"Active speakers: {summary['active_speaker_count']}")
    print(f"Seed: {summary['assignment_seed']}")
    print(f"Wrote: {args.output}")
    print(f"Summary: {args.summary}")
    print(f"First assignment: {assignments[0]['role_assignments']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
