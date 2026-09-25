"""Freeze persistent corpus identities onto the approved Higgs v0.2 pool.

Run from the repository root with ``uv run python -m scripts.assign_production_speakers``.
No TTS is performed. Rendering subsets must consume the frozen output map.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "materialize_speaker_assignments",
    ROOT / "scripts/materialize_speaker_assignments.py",
)
assert _SPEC is not None and _SPEC.loader is not None
materializer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = materializer
_SPEC.loader.exec_module(materializer)
input_helpers = materializer.assign

INPUT = Path("data/final/corpus_v1_1000.jsonl")
REGISTRY = Path("data/speaker_pool/vctk_v0.2/speaker_registry.json")
ACTIVE = Path("data/speaker_pool/vctk_v0.2/active_speakers.json")
OUTPUT = Path("data/speaker_assignment/v0.2")
POOL_VERSION = "vctk_v0.2"
POLICY_VERSION = "speaker_assignment_policy_v0.2"
SEED = 5703
ROLES = ("caller", "counsellor")
GENDERS = ("female", "male", "neutral")
SOURCE_GENDERS = {"female": "F", "male": "M"}
PARTITION = {
    "caller": {
        "F": ("spk_001", "spk_003", "spk_004", "spk_014", "spk_015"),
        "M": ("spk_008", "spk_009", "spk_010", "spk_011"),
    },
    "counsellor": {
        "F": ("spk_006", "spk_016", "spk_017"),
        "M": ("spk_007", "spk_012", "spk_013"),
    },
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_voices(
    registry_path: Path = ROOT / REGISTRY,
    active_path: Path = ROOT / ACTIVE,
) -> dict[str, dict[str, Any]]:
    pool = input_helpers.load_and_validate_active_pool(active_path, registry_path)
    registry = input_helpers.load_registry(registry_path)
    partition_ids = [
        voice for role in ROLES for sex in ("F", "M") for voice in PARTITION[role][sex]
    ]
    if (
        pool.speaker_pool_version != POOL_VERSION
        or pool.exclusions
        or len(partition_ids) != len(set(partition_ids))
        or set(partition_ids) != set(pool.active_speaker_ids)
        or len(partition_ids) != 15
    ):
        raise ValueError(
            "v0.2 active voices must equal the disjoint 15-voice partition"
        )
    by_id = {entry["speaker_id"]: entry for entry in registry["speakers"]}
    if set(by_id) != set(partition_ids):
        raise ValueError("v0.2 registry must contain exactly the 15 active voices")

    voices = {}
    for role in ROLES:
        for sex in ("F", "M"):
            for speaker_id in PARTITION[role][sex]:
                entry = by_id[speaker_id]
                metadata = entry["source_metadata"]
                if metadata["gender"] != sex:
                    raise ValueError(
                        f"{speaker_id} source gender conflicts with partition"
                    )
                materializer._resolve_higgs_reference(
                    speaker_id, entry, project_root=ROOT
                )
                voices[speaker_id] = {
                    "role": role,
                    "source_speaker_id": entry["source_speaker_id"],
                    "source_gender": sex,
                    "source_age": int(metadata["age"]),
                }
    return voices


def collect_exposure(
    dialogues: list[input_helpers.DialogueRecord],
) -> dict[str, dict[str, Any]]:
    exposure: dict[str, dict[str, Any]] = {}
    for dialogue in dialogues:
        specs = dialogue.raw["scenario"]["speakers"]
        if set(dialogue.roles) != set(ROLES):
            raise ValueError(f"{dialogue.dialogue_id} must contain both logical roles")
        if specs["caller"]["speaker_id"] == specs["counsellor"]["speaker_id"]:
            raise ValueError(
                f"{dialogue.dialogue_id} reuses one upstream ID across roles"
            )
        for role in ROLES:
            spec = specs[role]
            upstream_id = spec["speaker_id"]
            requested_gender = spec["gender"]
            requested_age_band = spec["age_band"]
            if requested_gender not in GENDERS or not isinstance(
                requested_age_band, str
            ):
                raise ValueError(f"Invalid speaker metadata for {upstream_id}")
            identity = exposure.setdefault(
                upstream_id,
                {
                    "upstream_speaker_id": upstream_id,
                    "role": role,
                    "requested_gender": requested_gender,
                    "requested_age_band": requested_age_band,
                    "dialogue_count": 0,
                    "turn_count": 0,
                },
            )
            if (
                identity["role"],
                identity["requested_gender"],
                identity["requested_age_band"],
            ) != (role, requested_gender, requested_age_band):
                raise ValueError(f"Unstable role/gender/age_band for {upstream_id}")
            identity["dialogue_count"] += 1
        for turn in dialogue.turns:
            upstream_id = specs[turn.speaker]["speaker_id"]
            exposure[upstream_id]["turn_count"] += 1
    return exposure


def validate_canonical(
    dialogues: list[input_helpers.DialogueRecord],
    exposure: dict[str, dict[str, Any]],
) -> None:
    callers = {f"C{number:03d}" for number in range(48)}
    counsellors = {f"L{number:03d}" for number in range(12)}
    if len(dialogues) != 1000 or len({item.dialogue_id for item in dialogues}) != 1000:
        raise ValueError("Canonical input must contain exactly 1000 unique dialogues")
    if (
        len(exposure) != 60
        or {key for key, value in exposure.items() if value["role"] == "caller"}
        != callers
        or {key for key, value in exposure.items() if value["role"] == "counsellor"}
        != counsellors
        or any(
            value["dialogue_count"] < 2 or value["turn_count"] < 1
            for value in exposure.values()
        )
        or sum(value["turn_count"] for value in exposure.values())
        != sum(len(dialogue.turns) for dialogue in dialogues)
    ):
        raise ValueError("Canonical upstream identities or turn exposure changed")


def tie_break(seed: int, upstream_id: str, speaker_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{upstream_id}\0{speaker_id}".encode()).digest()


def assign_identity_map(
    exposure: dict[str, dict[str, Any]],
    voices: dict[str, dict[str, Any]],
    seed: int = SEED,
) -> dict[str, dict[str, Any]]:
    turn_load = Counter({speaker_id: 0 for speaker_id in voices})
    identity_load = Counter({speaker_id: 0 for speaker_id in voices})
    neutral_turns = {role: Counter({"F": 0, "M": 0}) for role in ROLES}
    assignments: dict[str, str] = {}

    for role in ROLES:
        for gender in ("female", "male", "neutral"):
            identities = sorted(
                (
                    item
                    for item in exposure.values()
                    if item["role"] == role and item["requested_gender"] == gender
                ),
                key=lambda item: (-item["turn_count"], item["upstream_speaker_id"]),
            )
            for identity in identities:
                upstream_id = identity["upstream_speaker_id"]
                if gender == "neutral":
                    lowest_neutral = min(neutral_turns[role].values())
                    eligible = [
                        speaker_id
                        for sex in ("F", "M")
                        if neutral_turns[role][sex] == lowest_neutral
                        for speaker_id in PARTITION[role][sex]
                    ]
                else:
                    eligible = list(PARTITION[role][SOURCE_GENDERS[gender]])
                chosen = min(
                    eligible,
                    key=lambda speaker_id: (
                        turn_load[speaker_id],
                        identity_load[speaker_id],
                        tie_break(seed, upstream_id, speaker_id),
                        speaker_id,
                    ),
                )
                assignments[upstream_id] = chosen
                turn_load[chosen] += identity["turn_count"]
                identity_load[chosen] += 1
                if gender == "neutral":
                    neutral_turns[role][voices[chosen]["source_gender"]] += identity[
                        "turn_count"
                    ]

    if set(assignments) != set(exposure):
        raise ValueError("Some upstream identities were not mapped")
    return {
        upstream_id: {
            **exposure[upstream_id],
            "render_speaker_id": assignments[upstream_id],
            "render_source_speaker_id": voices[assignments[upstream_id]][
                "source_speaker_id"
            ],
            "render_source_gender": voices[assignments[upstream_id]]["source_gender"],
            "render_source_age": voices[assignments[upstream_id]]["source_age"],
        }
        for upstream_id in sorted(assignments)
    }


def assignment_rows(
    dialogues: list[input_helpers.DialogueRecord],
    frozen_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for dialogue in sorted(dialogues, key=lambda item: item.dialogue_id):
        specs = dialogue.raw["scenario"]["speakers"]
        roles = {}
        for role in ROLES:
            upstream_id = specs[role]["speaker_id"]
            mapping = frozen_map[upstream_id]
            if (
                mapping["role"] != role
                or mapping["requested_gender"] != specs[role]["gender"]
                or mapping["requested_age_band"] != specs[role]["age_band"]
            ):
                raise ValueError(
                    f"Frozen mapping disagrees with {dialogue.dialogue_id} {role}"
                )
            roles[role] = mapping["render_speaker_id"]
        if roles["caller"] == roles["counsellor"]:
            raise ValueError(
                f"{dialogue.dialogue_id} uses one render voice for both roles"
            )
        rows.append(
            {
                "dialogue_id": dialogue.dialogue_id,
                "speaker_pool_version": POOL_VERSION,
                "assignment_policy_version": POLICY_VERSION,
                "role_assignments": roles,
            }
        )
    return rows


def counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "identity_count": len(items),
        "dialogue_count": sum(item["dialogue_count"] for item in items),
        "turn_count": sum(item["turn_count"] for item in items),
    }


def build_summary(
    dialogues: list[input_helpers.DialogueRecord],
    frozen_map: dict[str, dict[str, Any]],
    voices: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    identities = list(frozen_map.values())
    per_voice = {
        speaker_id: {
            **voices[speaker_id],
            **counts(
                [item for item in identities if item["render_speaker_id"] == speaker_id]
            ),
        }
        for speaker_id in sorted(voices)
    }
    neutral = [item for item in identities if item["requested_gender"] == "neutral"]
    source_ages = sorted({voice["source_age"] for voice in voices.values()})
    caller_voices = set(PARTITION["caller"]["F"] + PARTITION["caller"]["M"])
    counsellor_voices = set(PARTITION["counsellor"]["F"] + PARTITION["counsellor"]["M"])
    return {
        "assignment_policy_version": POLICY_VERSION,
        "speaker_pool_version": POOL_VERSION,
        "corpus": {
            "dialogue_count": len(dialogues),
            "turn_count": sum(len(dialogue.turns) for dialogue in dialogues),
            "caller_upstream_identity_count": sum(
                item["role"] == "caller" for item in identities
            ),
            "counsellor_upstream_identity_count": sum(
                item["role"] == "counsellor" for item in identities
            ),
        },
        "mapping": {
            "upstream_identity_count": len(identities),
            "upstream_speaker_ids": sorted(frozen_map),
            "per_render_voice": per_voice,
            "all_15_voices_used": all(
                item["identity_count"] > 0 for item in per_voice.values()
            ),
            "role_exposure": {
                role: counts([item for item in identities if item["role"] == role])
                for role in ROLES
            },
        },
        "gender": {
            "requested": {
                gender: counts(
                    [item for item in identities if item["requested_gender"] == gender]
                )
                for gender in GENDERS
            },
            "neutral_to_source": {
                sex: counts(
                    [item for item in neutral if item["render_source_gender"] == sex]
                )
                for sex in ("F", "M")
            },
            "neutral_to_source_by_role": {
                role: {
                    sex: counts(
                        [
                            item
                            for item in neutral
                            if item["role"] == role
                            and item["render_source_gender"] == sex
                        ]
                    )
                    for sex in ("F", "M")
                }
                for role in ROLES
            },
        },
        "role_partition": {
            "caller_voices": sorted(caller_voices),
            "counsellor_voices": sorted(counsellor_voices),
            "intersection": sorted(caller_voices & counsellor_voices),
        },
        "age": {
            "requested_age_band": {
                band: counts(
                    [item for item in identities if item["requested_age_band"] == band]
                )
                for band in sorted({item["requested_age_band"] for item in identities})
            },
            "render_source_age": {
                str(age): {
                    "source_voice_count": sum(
                        voice["source_age"] == age for voice in voices.values()
                    ),
                    **counts(
                        [
                            item
                            for item in identities
                            if item["render_source_age"] == age
                        ]
                    ),
                }
                for age in source_ages
            },
            "note": "Requested age_band and VCTK source age are descriptive; source age is not age-band realization or a selection input.",
        },
    }


def build_artifacts(
    input_path: Path = ROOT / INPUT,
    registry_path: Path = ROOT / REGISTRY,
    active_path: Path = ROOT / ACTIVE,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    voices = load_voices(registry_path, active_path)
    dialogues = input_helpers.load_dialogues(input_path)
    exposure = collect_exposure(dialogues)
    validate_canonical(dialogues, exposure)
    frozen_map = assign_identity_map(exposure, voices)
    if set(voices) != {item["render_speaker_id"] for item in frozen_map.values()}:
        raise ValueError("All 15 production voices must serve at least one identity")
    rows = assignment_rows(dialogues, frozen_map)
    if len(rows) != 1000 or len({row["dialogue_id"] for row in rows}) != 1000:
        raise ValueError("Expected exactly one assignment row per canonical dialogue")
    policy = {
        "assignment_policy_version": POLICY_VERSION,
        "speaker_pool_version": POOL_VERSION,
        "policy_seed": SEED,
        "canonical_input": {
            "path": str(input_path.relative_to(ROOT)),
            "sha256": sha256(input_path),
        },
        "speaker_registry": {
            "path": str(registry_path.relative_to(ROOT)),
            "sha256": sha256(registry_path),
        },
        "active_speakers": {
            "path": str(active_path.relative_to(ROOT)),
            "sha256": sha256(active_path),
        },
        "fixed_role_partition": PARTITION,
        "gender_policy": "Requested female requires source F; requested male requires source M; requested neutral permits source F or M and remains neutral metadata.",
        "neutral_policy": "Per role, prefer lower neutral turn exposure by source gender, then lower total voice turn load, identity count, and SHA256 seed tie-break.",
        "age_policy": "Requested age_band is audit-only; VCTK source age is descriptive, not age-band realization, and does not select a voice.",
        "identity_map": frozen_map,
    }
    summary = build_summary(dialogues, frozen_map, voices)
    return policy, rows, summary


def write_artifacts(
    policy: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path = ROOT / OUTPUT,
) -> None:
    paths = [
        output_dir / name
        for name in ("policy.json", "speaker_assignments.jsonl", "summary.json")
    ]
    if any(path.exists() for path in paths):
        raise FileExistsError(
            f"Refusing to overwrite frozen assignment artifacts in {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(
        json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths[1].write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    paths[2].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    policy, rows, summary = build_artifacts()
    write_artifacts(policy, rows, summary)
    print(
        f"Frozen {len(policy['identity_map'])} identities across {len(rows)} dialogues in {ROOT / OUTPUT}"
    )


if __name__ == "__main__":
    main()
