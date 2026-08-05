from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from medical_triage_agent.contracts import DPOExample, Metadata, SFTExample, content_fingerprint
from medical_triage_agent.dataset_audit import audit_records
from medical_triage_agent.normalize import assign_split, normalize_dpo_record, normalize_sft_record
from medical_triage_agent.source_registry import (
    SourceRecord,
    load_source_registry,
    validate_source_for_use,
)

DatasetKind = Literal["sft", "dpo"]
SPLITS = ("train", "validation", "test", "clinical_eval")


@dataclass(frozen=True)
class GeneratedDataset:
    manifest_path: Path
    split_paths: tuple[Path, ...]
    card_path: Path | None = None


@dataclass(frozen=True)
class TrainingDataConfig:
    output_dir: Path
    sft_target: int = 5000
    dpo_target: int = 1000
    sft_mediqa: int = 2000
    sft_frenchmedmcqa: int = 1000


def load_hf_sft_records(source_id: str, limit: int | None = None) -> list[SFTExample]:
    registry = load_source_registry()
    source = validate_source_for_use(source_id, "sft", registry)

    if source_id == "mediqa":
        dataset = _load_hf_dataset("ANR-MALADES/MediQAl", "oeq", split="test")
        rows = _limit(dataset, limit)
        return [map_mediqa_oeq(row, source) for row in rows]
    if source_id == "frenchmedmcqa":
        dataset = _load_hf_dataset("nthngdy/frenchmedmcqa", split="train")
        rows = _limit(dataset, limit)
        return [map_frenchmedmcqa(row, source) for row in rows]
    if source_id == "medquad":
        dataset = _load_hf_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
        rows = _limit(dataset, limit)
        return [map_medquad(row, source) for row in rows]
    raise ValueError(f"no SFT mapper is implemented for source: {source_id}")


def map_frenchmedmcqa(row: dict[str, Any], source: SourceRecord) -> SFTExample:
    question = _string(row, "question")
    options = _options(row)
    correct_index = _int(row, "correct_answers")
    if correct_index not in range(len(options)):
        raise ValueError("correct_answers must point to an existing option")
    answer_label = chr(ord("A") + correct_index)
    answer = options[correct_index]

    # ponytail: exam MCQA data has no CHSA triage label; default to moderee until clinician
    # triage labels exist.
    metadata = Metadata(
        symptoms=[],
        antecedents=[],
        vitals={},
        triage_level="moderee",
        confidence=0.4,
        source=source.id,
        license=source.license,
        transforms=["load_hf:nthngdy/frenchmedmcqa:train", "map_frenchmedmcqa", "redact_pii"],
    )
    formatted_options = "\n".join(
        f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(options)
    )
    return normalize_sft_record(
        language="fr",
        instruction="Choisir la bonne reponse au QCM medical et expliquer brievement.",
        input_text=f"Question:\n{question}\n\nOptions:\n{formatted_options}",
        output_text=f"Reponse {answer_label}: {answer}",
        source_id=source.id,
        metadata=metadata,
    )


def map_medquad(row: dict[str, Any], source: SourceRecord) -> SFTExample:
    question = _string(row, "Question")
    answer = _string(row, "Answer")
    qtype = str(row.get("qtype", "unknown"))

    # ponytail: general medical QA is not triage-labeled; default to moderee until clinician
    # triage labels exist.
    metadata = Metadata(
        symptoms=[],
        antecedents=[],
        vitals={},
        triage_level="moderee",
        confidence=0.4,
        source=source.id,
        license=source.license,
        transforms=[
            "load_hf:keivalya/MedQuad-MedicalQnADataset:train",
            "map_medquad",
            "redact_pii",
        ],
    )
    return normalize_sft_record(
        language="en",
        instruction=f"Answer the medical question. Category: {qtype}.",
        input_text=question,
        output_text=answer,
        source_id=source.id,
        metadata=metadata,
    )


def load_hf_dpo_records(source_id: str, limit: int | None = None) -> list[DPOExample]:
    registry = load_source_registry()
    source = validate_source_for_use(source_id, "dpo", registry)
    if source_id != "ultramedical_preference":
        raise ValueError(f"no DPO mapper is implemented for source: {source_id}")

    dataset = _load_hf_dataset("TsinghuaC3I/UltraMedical-Preference", split="train")
    rows = _limit(dataset, limit)
    return [map_ultramedical_preference(row, source) for row in rows]


def map_mediqa_oeq(row: dict[str, Any], source: SourceRecord) -> SFTExample:
    clinical_case = _string(row, "clinical_case")
    question = _string(row, "question")
    answer = _string(row, "answer")
    # ponytail: public QA data is not triage-labeled; default to moderee until clinician labels
    # are added or a real triage dataset is verified.
    metadata = Metadata(
        symptoms=[],
        antecedents=[],
        vitals={},
        triage_level="moderee",
        confidence=0.4,
        source=source.id,
        license=source.license,
        transforms=["load_hf:ANR-MALADES/MediQAl:oeq:test", "map_mediqa_oeq", "redact_pii"],
    )
    return normalize_sft_record(
        language="fr",
        instruction="Repondre a la question medicale a partir du cas clinique fourni.",
        input_text=f"Cas clinique:\n{clinical_case}\n\nQuestion:\n{question}",
        output_text=answer,
        source_id=source.id,
        metadata=metadata,
    )


def map_ultramedical_preference(row: dict[str, Any], source: SourceRecord) -> DPOExample:
    prompt = _string(row, "prompt")
    chosen = _last_assistant_content(row.get("chosen"))
    rejected = _last_assistant_content(row.get("rejected"))

    # ponytail: public preference data is not triage-labeled; default to moderee until clinician
    # preference pairs include CHSA triage labels.
    metadata = Metadata(
        symptoms=[],
        antecedents=[],
        vitals={},
        triage_level="moderee",
        confidence=0.4,
        source=source.id,
        license=source.license,
        transforms=[
            "load_hf:TsinghuaC3I/UltraMedical-Preference:train",
            "map_ultramedical_preference",
            "redact_pii",
        ],
    )
    return normalize_dpo_record(
        language="en",
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        source_id=source.id,
        metadata=metadata,
    )


def write_splits(
    records: Iterable[SFTExample | DPOExample | dict[str, Any]],
    output_dir: Path,
    kind: DatasetKind,
) -> GeneratedDataset:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for record in records:
        example = _coerce_record(record, kind)
        # Split from the stable record ID so reruns keep train/eval boundaries identical.
        split_rows[assign_split(example.id)].append(example.to_dict())

    split_paths: list[Path] = []
    for split_name, rows in split_rows.items():
        path = output_dir / f"{kind}_{split_name}.jsonl"
        _write_jsonl(path, rows)
        split_paths.append(path)
        result = audit_records(rows, kind)
        if not result.ok:
            raise ValueError(f"{path} failed audit: {'; '.join(result.errors)}")

    manifest_path = output_dir / "manifest.json"
    manifest = build_manifest(kind, split_rows)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return GeneratedDataset(manifest_path=manifest_path, split_paths=tuple(split_paths))


def build_training_data(config: TrainingDataConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Build SFT and DPO together so the manifest describes one coherent training snapshot.
    sft_records = _build_sft_records(config)
    dpo_records = load_hf_dpo_records("ultramedical_preference", config.dpo_target)

    sft_rows = _split_examples(sft_records)
    dpo_rows = _split_examples(dpo_records)
    sft_paths = _write_kind_splits(config.output_dir, "sft", sft_rows)
    dpo_paths = _write_kind_splits(config.output_dir, "dpo", dpo_rows)

    audit_report = build_audit_report(sft_rows, dpo_rows)
    audit_report_path = config.output_dir / "audit_report.json"
    audit_report_path.write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    review_queue_path = config.output_dir / "clinical_review_queue.jsonl"
    review_rows = build_clinical_review_queue(sft_rows, dpo_rows)
    _write_jsonl(review_queue_path, review_rows)

    manifest = build_training_manifest(
        config=config,
        sft_rows=sft_rows,
        dpo_rows=dpo_rows,
        audit_report_path=audit_report_path,
        clinical_review_queue_path=review_queue_path,
        split_paths=(*sft_paths, *dpo_paths),
    )
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    make_dataset_card(manifest_path, config.output_dir / "README.md")
    return manifest


def audit_training_data(output_dir: Path) -> dict[str, Any]:
    sft_rows = _read_kind_splits(output_dir, "sft")
    dpo_rows = _read_kind_splits(output_dir, "dpo")
    audit_report = build_audit_report(sft_rows, dpo_rows)
    manifest_path = output_dir / "manifest.json"
    # These files are part of the release artifact contract, not nice-to-have sidecars.
    if not manifest_path.exists():
        audit_report["errors"].append("missing manifest.json")
    if not (output_dir / "clinical_review_queue.jsonl").exists():
        audit_report["errors"].append("missing clinical_review_queue.jsonl")
    if not (output_dir / "README.md").exists():
        audit_report["errors"].append("missing dataset card README.md")
    return audit_report


def summarize_training_data(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "output_dir": str(output_dir),
        "actual_counts": manifest["actual_counts"],
        "language_counts": manifest["language_counts"],
        "source_counts": manifest["source_counts"],
        "split_counts": manifest["split_counts"],
        "rejected_counts": manifest["rejected_counts"],
        "files": manifest["files"],
    }


def build_manifest(
    kind: DatasetKind, split_rows: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    all_rows = [row for rows in split_rows.values() for row in rows]
    return {
        "kind": kind,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_ids": sorted({source_id for row in all_rows for source_id in row["source_ids"]}),
        "split_counts": {name: len(rows) for name, rows in split_rows.items()},
        "content_hashes": {name: _rows_hash(rows) for name, rows in split_rows.items()},
        "transforms": sorted(
            {transform for row in all_rows for transform in row["metadata"].get("transforms", [])}
        ),
        "record_fingerprints": sorted(content_fingerprint(row) for row in all_rows),
    }


def build_training_manifest(
    *,
    config: TrainingDataConfig,
    sft_rows: dict[str, list[dict[str, Any]]],
    dpo_rows: dict[str, list[dict[str, Any]]],
    audit_report_path: Path,
    clinical_review_queue_path: Path,
    split_paths: tuple[Path, ...],
) -> dict[str, Any]:
    all_rows = _all_rows(sft_rows, dpo_rows)
    # The manifest is intentionally redundant: it should be enough to audit a dataset folder
    # without replaying the whole ingestion job.
    return {
        "kind": "training_data",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "target_counts": {
            "sft": config.sft_target,
            "dpo": config.dpo_target,
            "sft_mediqa": config.sft_mediqa,
            "sft_frenchmedmcqa": config.sft_frenchmedmcqa,
        },
        "actual_counts": {"sft": len(_all_rows(sft_rows)), "dpo": len(_all_rows(dpo_rows))},
        "language_counts": _count_by(all_rows, "language"),
        "source_counts": _source_counts(all_rows),
        "split_counts": {
            "sft": {split: len(rows) for split, rows in sft_rows.items()},
            "dpo": {split: len(rows) for split, rows in dpo_rows.items()},
        },
        "content_hashes": {
            "sft": {split: _rows_hash(rows) for split, rows in sft_rows.items()},
            "dpo": {split: _rows_hash(rows) for split, rows in dpo_rows.items()},
        },
        "transforms": sorted(
            {transform for row in all_rows for transform in row["metadata"].get("transforms", [])}
        ),
        "rejected_counts": {"sft": 0, "dpo": 0},
        "audit_report_path": str(audit_report_path),
        "clinical_review_queue_path": str(clinical_review_queue_path),
        "files": [str(path) for path in split_paths],
    }


def build_audit_report(
    sft_rows: dict[str, list[dict[str, Any]]],
    dpo_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    errors: list[str] = []
    # Reuse row-level audits here; this wrapper adds cross-split and artifact-level signals.
    sft_errors = _audit_kind_rows(sft_rows, "sft")
    dpo_errors = _audit_kind_rows(dpo_rows, "dpo")
    errors.extend(sft_errors)
    errors.extend(dpo_errors)
    errors.extend(_split_overlap_errors(sft_rows, "sft"))
    errors.extend(_split_overlap_errors(dpo_rows, "dpo"))
    return {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "passed": not errors,
        "errors": errors,
        "accepted_counts": {
            "sft": len(_all_rows(sft_rows)),
            "dpo": len(_all_rows(dpo_rows)),
        },
        "rejected_counts": {"sft": 0, "dpo": 0},
        "pii_findings": sum("possible PII" in error for error in errors),
        "duplicate_findings": sum("duplicate" in error for error in errors),
        "missing_provenance_findings": sum("source" in error for error in errors),
        "language_counts": _count_by(_all_rows(sft_rows, dpo_rows), "language"),
        "source_counts": _source_counts(_all_rows(sft_rows, dpo_rows)),
        "split_counts": {
            "sft": {split: len(rows) for split, rows in sft_rows.items()},
            "dpo": {split: len(rows) for split, rows in dpo_rows.items()},
        },
        "transforms": sorted(
            {
                transform
                for row in _all_rows(sft_rows, dpo_rows)
                for transform in row["metadata"].get("transforms", [])
            }
        ),
    }


def build_clinical_review_queue(
    sft_rows: dict[str, list[dict[str, Any]]],
    dpo_rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind, split_rows in (("sft", sft_rows), ("dpo", dpo_rows)):
        for split, records in split_rows.items():
            for record in records:
                metadata = record["metadata"]
                transforms = metadata.get("transforms", [])
                # Public QA/preference data teaches format and medical language, not CHSA triage
                # policy; the queue makes that validation debt explicit.
                if metadata.get("triage_level") == "urgence_maximale" or _needs_review(transforms):
                    rows.append(
                        {
                            "id": record["id"],
                            "kind": kind,
                            "split": split,
                            "language": record["language"],
                            "source_ids": record["source_ids"],
                            "triage_level": metadata.get("triage_level"),
                            "reason": "public medical QA/preference record requires clinician review",
                        }
                    )
    return rows


def make_dataset_card(manifest_path: Path, output_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines = [
        "# CHSA medical triage POC dataset",
        "",
        "## Dataset summary",
        "",
        f"Kind: `{manifest['kind']}`.",
        f"Generated at: `{manifest['generated_at']}`.",
        "Sources: " + ", ".join(f"`{source}`" for source in _manifest_sources(manifest)) + ".",
        "",
        "## Split sizes",
        "",
        "| split | records |",
        "| --- | ---: |",
    ]
    lines.extend(_split_count_lines(manifest))
    lines.extend(
        [
            "",
            "## Processing",
            "",
            "Transforms: " + ", ".join(f"`{item}`" for item in manifest["transforms"]) + ".",
            "",
            "## Intended use",
            "",
            "This dataset is intended for the CHSA medical triage POC only and must not be used as a standalone diagnostic dataset.",
            "",
            "## Limitations",
            "",
            "The current public sources are medical QA/preference datasets, not clinician-validated CHSA triage records.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _build_sft_records(config: TrainingDataConfig) -> list[SFTExample]:
    mediqa_target = min(config.sft_mediqa, config.sft_target)
    french_target = min(config.sft_frenchmedmcqa, max(0, config.sft_target - mediqa_target))
    records = [
        *load_hf_sft_records("mediqa", mediqa_target),
        *load_hf_sft_records("frenchmedmcqa", french_target),
    ]
    remaining = max(0, config.sft_target - len(records))
    if remaining:
        # MedQuad is the English filler source; this keeps the target size stable when French
        # sources are smaller than requested.
        records.extend(load_hf_sft_records("medquad", remaining))
    return records[: config.sft_target]


def _split_examples(records: Iterable[SFTExample | DPOExample]) -> dict[str, list[dict[str, Any]]]:
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for record in records:
        split_rows[assign_split(record.id)].append(record.to_dict())
    return split_rows


def _write_kind_splits(
    output_dir: Path, kind: DatasetKind, split_rows: dict[str, list[dict[str, Any]]]
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for split in SPLITS:
        path = output_dir / f"{kind}_{split}.jsonl"
        _write_jsonl(path, split_rows[split])
        paths.append(path)
    return tuple(paths)


def _read_kind_splits(output_dir: Path, kind: DatasetKind) -> dict[str, list[dict[str, Any]]]:
    from medical_triage_agent.contracts import load_jsonl

    # Read every split explicitly so a missing file fails the audit instead of being ignored.
    return {split: load_jsonl(str(output_dir / f"{kind}_{split}.jsonl")) for split in SPLITS}


def _audit_kind_rows(split_rows: dict[str, list[dict[str, Any]]], kind: DatasetKind) -> list[str]:
    errors: list[str] = []
    for split, rows in split_rows.items():
        result = audit_records(rows, kind)
        errors.extend(f"{kind}_{split}: {error}" for error in result.errors)
    return errors


def _split_overlap_errors(
    split_rows: dict[str, list[dict[str, Any]]], kind: DatasetKind
) -> list[str]:
    owners: dict[str, str] = {}
    errors: list[str] = []
    for split, rows in split_rows.items():
        for row in rows:
            record_id = row["id"]
            if record_id in owners:
                errors.append(
                    f"{kind}: record {record_id} appears in {owners[record_id]} and {split}"
                )
            owners[record_id] = split
    return errors


def _all_rows(*split_groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for group in split_groups for rows in group.values() for row in rows]


def _count_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _source_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for source_id in row["source_ids"]:
            counts[source_id] = counts.get(source_id, 0) + 1
    return dict(sorted(counts.items()))


def _needs_review(transforms: Any) -> bool:
    if not isinstance(transforms, list):
        return True
    review_transforms = ("map_mediqa_oeq", "map_frenchmedmcqa", "map_medquad")
    return any(transform in transforms for transform in review_transforms)


def _manifest_sources(manifest: dict[str, Any]) -> list[str]:
    if "source_ids" in manifest:
        return list(manifest["source_ids"])
    return list(manifest.get("source_counts", {}).keys())


def _split_count_lines(manifest: dict[str, Any]) -> list[str]:
    split_counts = manifest["split_counts"]
    if "sft" in split_counts or "dpo" in split_counts:
        lines: list[str] = []
        for kind, counts in split_counts.items():
            lines.extend(f"| {kind}_{split} | {count} |" for split, count in counts.items())
        return lines
    return [f"| {split} | {count} |" for split, count in split_counts.items()]


def _limit(rows: Iterable[dict[str, Any]], limit: int | None) -> Iterable[dict[str, Any]]:
    if limit is None:
        return rows
    return (row for index, row in enumerate(rows) if index < limit)


def _load_hf_dataset(*args: Any, **kwargs: Any) -> Iterable[dict[str, Any]]:
    # Keep HF tooling out of the default install; data builds opt into the heavier stack.
    try:
        datasets = importlib.import_module("datasets")
    except ImportError as exc:
        raise RuntimeError("install training extras with `uv sync --extra training`") from exc
    loaded = datasets.load_dataset(*args, **kwargs)
    if not isinstance(loaded, Iterable):
        raise TypeError("loaded Hugging Face dataset must be iterable")
    return loaded


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _rows_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _options(row: dict[str, Any]) -> list[str]:
    return [_string(row, f"answer_{label}") for label in ("a", "b", "c", "d", "e")]


def _last_assistant_content(messages: Any) -> str:
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if not isinstance(content, str):
                continue
            return content
    raise ValueError("messages must include assistant content")


def _coerce_record(
    record: SFTExample | DPOExample | dict[str, Any], kind: DatasetKind
) -> SFTExample | DPOExample:
    if isinstance(record, SFTExample | DPOExample):
        return record
    return SFTExample.from_mapping(record) if kind == "sft" else DPOExample.from_mapping(record)
