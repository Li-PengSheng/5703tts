"""Build a derived acceptance index from canonical records and batch manifest v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from tts5703.input.records import InputRecordFailure, read_input_records

SCHEMA_VERSION = "1.0"
BATCH_VERSION = "2.0"
STATUSES = (
    "accepted_first_pass",
    "accepted_after_retry",
    "quality_rejected_pending",
    "failed_integrity",
    "failed_render_or_input",
    "excluded_known_issue",
    "not_attempted",
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _codes(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, dict) or not isinstance(value.get("issues"), list):
        raise TypeError(f"Invalid {field}")
    codes = []
    for issue in value["issues"]:
        if (
            not isinstance(issue, dict)
            or not isinstance(issue.get("code"), str)
            or not issue["code"].strip()
        ):
            raise ValueError(f"Invalid {field}.issues")
        codes.append(issue["code"])
    return sorted(set(codes))


def _attempt(
    result: dict[str, Any], backend: str, label: str, kind: str
) -> tuple[str, dict[str, Any]]:
    action, status = result.get("action"), result.get("status")
    quality = result.get("quality")
    observation = result.get("quality_observation")
    quality_codes = _codes(quality, "quality")
    observation_codes = _codes(observation, "quality_observation")
    outcome = quality.get("outcome") if isinstance(quality, dict) else None
    if quality is not None and outcome not in {"pass", "reject", "indeterminate"}:
        raise ValueError(f"Invalid quality outcome in {label}")
    if action in {"rendered", "resumed"} and status == "success":
        disposition = (
            "failed_integrity"
            if result.get("error") is not None
            or observation is not None
            or quality_codes
            or (backend == "higgs" and outcome != "pass")
            or outcome in {"reject", "indeterminate"}
            else "accepted"
        )
    elif action == "quality_rejected" and status == "failed":
        disposition = (
            "quality_rejected_pending"
            if outcome == "reject"
            and observation is None
            and "artifact_integrity_failure" not in quality_codes
            and isinstance(result.get("error"), dict)
            and result["error"].get("type") == "QualityRejected"
            else "failed_integrity"
        )
    elif action in {"render_failed", "input_error"} and status == "failed":
        disposition = "failed_render_or_input"
    elif action == "excluded_known_issue" and status == "excluded":
        disposition = "excluded_known_issue"
    else:
        raise ValueError(f"Invalid action/status in {label}: {action!r}/{status!r}")
    attempt = {
        "source_label": label,
        "source_kind": kind,
        "action": action,
        "status": status,
        "disposition": disposition,
        "quality_outcome": outcome,
        "quality_reason_codes": quality_codes,
        "quality_observation_reason_codes": observation_codes,
    }
    for field in ("render_fingerprint", "output_dir"):
        value = result.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Invalid {field} in {label}")
        attempt[field] = value
    return disposition, attempt


def build_manifest(
    corpus: Path, sources: list[tuple[str, str, Path]]
) -> dict[str, Any]:
    corpus_bytes = corpus.read_bytes()
    records = read_input_records(corpus)
    canonical = {}
    for record in records:
        if isinstance(record, InputRecordFailure):
            raise TypeError(
                f"Invalid canonical corpus at {record.container_path}:{record.line_number}: {record.message}"
            )
        if record.dialogue_id in canonical:
            raise ValueError(f"Duplicate canonical dialogue: {record.dialogue_id}")
        canonical[record.dialogue_id] = record.record_sha256
    if not canonical:
        raise ValueError("Canonical corpus is empty")
    histories: dict[str, list[dict[str, Any]]] = {
        dialogue_id: [] for dialogue_id in canonical
    }
    provenance = []
    seen_labels = set()
    first_pass_ids = set()
    loaded = []
    for kind, label, path in sources:
        if (
            kind not in {"first_pass", "retry"}
            or not label
            or not label.strip()
            or label in seen_labels
        ):
            raise ValueError(f"Invalid or duplicate source label: {label!r}")
        seen_labels.add(label)
        data = path.read_bytes()
        manifest = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if (
            not isinstance(manifest, dict)
            or manifest.get("manifest_version") != BATCH_VERSION
        ):
            raise ValueError(f"Unsupported batch manifest version: {path}")
        if manifest.get("backend") not in {"higgs", "cosyvoice"} or not isinstance(
            manifest.get("results"), list
        ):
            raise ValueError(f"Invalid batch manifest: {path}")
        provenance.append(
            {
                "label": label,
                "kind": kind,
                "path": str(path),
                "sha256": hashlib.sha256(data).hexdigest(),
                "manifest_version": BATCH_VERSION,
            }
        )
        ids = set()
        for result in manifest["results"]:
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("dialogue_id"), str)
                or not result["dialogue_id"].strip()
            ):
                raise ValueError(f"Invalid dialogue result in {path}")
            dialogue_id = result["dialogue_id"]
            if dialogue_id in ids:
                raise ValueError(f"Duplicate dialogue in {path}: {dialogue_id}")
            ids.add(dialogue_id)
            if dialogue_id not in canonical:
                raise ValueError(f"Unknown dialogue in {path}: {dialogue_id}")
            source = result.get("source")
            if (
                not isinstance(source, dict)
                or source.get("record_sha256") != canonical[dialogue_id]
            ):
                raise ValueError(
                    f"Canonical source SHA mismatch in {path}: {dialogue_id}"
                )
        if kind == "first_pass":
            overlap = ids & first_pass_ids
            if overlap:
                raise ValueError(f"Overlapping first-pass dialogue: {sorted(overlap)}")
            first_pass_ids.update(ids)
        loaded.append((kind, label, manifest))
    for kind in ("first_pass", "retry"):
        for source_kind, label, manifest in loaded:
            if source_kind != kind:
                continue
            for result in manifest["results"]:
                dialogue_id = result["dialogue_id"]
                history = histories[dialogue_id]
                if kind == "retry" and (
                    not history
                    or history[-1]["disposition"]
                    in {"accepted", "excluded_known_issue"}
                ):
                    raise ValueError(
                        f"Retry lacks prior failed history or supersedes acceptance: {dialogue_id}"
                    )
                _, attempt = _attempt(result, manifest["backend"], label, kind)
                history.append(attempt)
    dialogues = []
    counts = Counter()
    for dialogue_id, record_sha256 in canonical.items():
        history = histories[dialogue_id]
        latest = history[-1] if history else None
        final_status = (
            (
                "accepted_after_retry"
                if latest["source_kind"] == "retry"
                else "accepted_first_pass"
            )
            if latest and latest["disposition"] == "accepted"
            else latest["disposition"]
            if latest
            else "not_attempted"
        )
        counts[final_status] += 1
        dialogues.append(
            {
                "dialogue_id": dialogue_id,
                "record_sha256": record_sha256,
                "final_status": final_status,
                "history": history,
                "accepted_source_label": latest["source_label"]
                if final_status.startswith("accepted_")
                else None,
                "accepted_source_kind": latest["source_kind"]
                if final_status.startswith("accepted_")
                else None,
                "accepted_render_fingerprint": latest["render_fingerprint"]
                if final_status.startswith("accepted_")
                else None,
                "accepted_output_dir": latest["output_dir"]
                if final_status.startswith("accepted_")
                else None,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_corpus": {
            "path": str(corpus),
            "sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        },
        "sources": provenance,
        "summary": {
            "dialogues_total": len(canonical),
            "accepted_total": counts["accepted_first_pass"]
            + counts["accepted_after_retry"],
            **{status: counts[status] for status in STATUSES},
        },
        "dialogues": dialogues,
    }


def write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(manifest, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class _SourceAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str,
        option_string: str | None = None,
    ) -> None:
        if "=" not in values:
            parser.error(f"{option_string} requires LABEL=PATH")
        label, path = values.split("=", 1)
        if not label.strip() or not path.strip():
            parser.error(f"{option_string} requires nonblank LABEL=PATH")
        sources = getattr(namespace, self.dest, None) or []
        sources.append(
            (
                "first_pass" if option_string == "--first-pass" else "retry",
                label,
                Path(path),
            )
        )
        setattr(namespace, self.dest, sources)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--first-pass", dest="sources", action=_SourceAction)
    parser.add_argument("--retry", dest="sources", action=_SourceAction)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.sources or not any(
            kind == "first_pass" for kind, _, _ in args.sources
        ):
            raise ValueError("At least one --first-pass source is required")
        if args.output.resolve() in {
            args.corpus.resolve(),
            *(path.resolve() for _, _, path in args.sources),
        }:
            raise ValueError("Output must differ from corpus and source manifests")
        manifest = build_manifest(args.corpus, args.sources)
        if (
            args.require_complete
            and manifest["summary"]["accepted_total"]
            != manifest["summary"]["dialogues_total"]
        ):
            raise ValueError("Canonical corpus is not fully accepted")
        write_manifest(args.output, manifest)
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        print(f"Acceptance manifest: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
