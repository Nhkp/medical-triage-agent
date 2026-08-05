from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medical_triage_agent.contracts import load_jsonl
from medical_triage_agent.dataset_audit import assert_split_isolation, audit_jsonl
from medical_triage_agent.dataset_pipeline import (
    TrainingDataConfig,
    audit_training_data,
    build_training_data,
    load_hf_dpo_records,
    load_hf_sft_records,
    make_dataset_card,
    summarize_training_data,
    write_splits,
)
from medical_triage_agent.evaluation import run_safety_evaluation
from medical_triage_agent.source_registry import load_source_registry


def main() -> int:
    parser = argparse.ArgumentParser(prog="medical-triage-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("sources", help="validate docs/data-sources.md")

    audit_parser = subparsers.add_parser("audit-jsonl", help="audit SFT or DPO JSONL")
    audit_parser.add_argument("kind", choices=["sft", "dpo"])
    audit_parser.add_argument("path")

    splits_parser = subparsers.add_parser("check-splits", help="verify split ID isolation")
    splits_parser.add_argument("paths", nargs="+")

    ingest_sft_parser = subparsers.add_parser("ingest-sft", help="load and split SFT records")
    ingest_sft_parser.add_argument("source_id")
    ingest_sft_parser.add_argument("output_dir")
    ingest_sft_parser.add_argument("--limit", type=int)

    ingest_dpo_parser = subparsers.add_parser("ingest-dpo", help="load and split DPO records")
    ingest_dpo_parser.add_argument("source_id")
    ingest_dpo_parser.add_argument("output_dir")
    ingest_dpo_parser.add_argument("--limit", type=int)

    write_splits_parser = subparsers.add_parser(
        "write-splits",
        help="assign an existing JSONL file to deterministic split files",
    )
    write_splits_parser.add_argument("kind", choices=["sft", "dpo"])
    write_splits_parser.add_argument("input_path")
    write_splits_parser.add_argument("output_dir")

    card_parser = subparsers.add_parser("make-dataset-card", help="render a dataset card")
    card_parser.add_argument("manifest_path")
    card_parser.add_argument("output_path")

    evaluation_parser = subparsers.add_parser("evaluate-safety", help="run v1 safety checks")
    evaluation_parser.add_argument("--output")

    build_training_parser = subparsers.add_parser(
        "build-training-data",
        help="build unified local SFT and DPO training data",
    )
    build_training_parser.add_argument("output_dir")
    build_training_parser.add_argument("--sft-target", type=int, default=5000)
    build_training_parser.add_argument("--dpo-target", type=int, default=1000)
    build_training_parser.add_argument("--sft-mediqa", type=int, default=2000)
    build_training_parser.add_argument("--sft-frenchmedmcqa", type=int, default=1000)

    audit_training_parser = subparsers.add_parser(
        "audit-training-data",
        help="audit unified local training data artifacts",
    )
    audit_training_parser.add_argument("output_dir")

    summary_parser = subparsers.add_parser(
        "summarize-training-data",
        help="summarize unified local training data artifacts",
    )
    summary_parser.add_argument("output_dir")

    args = parser.parse_args()
    if args.command == "sources":
        sources = load_source_registry()
        print(f"loaded {len(sources)} sources")
        return 0
    if args.command == "audit-jsonl":
        result = audit_jsonl(args.path, args.kind)
        for error in result.errors:
            print(error, file=sys.stderr)
        print(f"checked {result.checked} records")
        return 0 if result.ok else 1
    if args.command == "check-splits":
        splits = {Path(path).stem: load_jsonl(path) for path in args.paths}
        assert_split_isolation(splits)
        print(f"checked {len(splits)} splits")
        return 0
    if args.command == "ingest-sft":
        dataset = write_splits(
            load_hf_sft_records(args.source_id, args.limit),
            Path(args.output_dir),
            "sft",
        )
        print(f"wrote {len(dataset.split_paths)} splits and {dataset.manifest_path}")
        return 0
    if args.command == "ingest-dpo":
        dataset = write_splits(
            load_hf_dpo_records(args.source_id, args.limit),
            Path(args.output_dir),
            "dpo",
        )
        print(f"wrote {len(dataset.split_paths)} splits and {dataset.manifest_path}")
        return 0
    if args.command == "write-splits":
        records = load_jsonl(args.input_path)
        dataset = write_splits(records, Path(args.output_dir), args.kind)
        print(f"wrote {len(dataset.split_paths)} splits and {dataset.manifest_path}")
        return 0
    if args.command == "make-dataset-card":
        path = make_dataset_card(Path(args.manifest_path), Path(args.output_path))
        print(f"wrote {path}")
        return 0
    if args.command == "evaluate-safety":
        safety_result = run_safety_evaluation()
        serialized = json.dumps(safety_result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(serialized, encoding="utf-8")
        else:
            print(serialized, end="")
        return 0 if safety_result["passed"] else 1
    if args.command == "build-training-data":
        manifest = build_training_data(
            TrainingDataConfig(
                output_dir=Path(args.output_dir),
                sft_target=args.sft_target,
                dpo_target=args.dpo_target,
                sft_mediqa=args.sft_mediqa,
                sft_frenchmedmcqa=args.sft_frenchmedmcqa,
            )
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit-training-data":
        audit_report = audit_training_data(Path(args.output_dir))
        print(json.dumps(audit_report, ensure_ascii=False, indent=2))
        return 0 if audit_report["passed"] else 1
    if args.command == "summarize-training-data":
        print(
            json.dumps(summarize_training_data(Path(args.output_dir)), ensure_ascii=False, indent=2)
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
