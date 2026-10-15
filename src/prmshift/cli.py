from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from .io import read_jsonl, write_json, write_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(description="Conditional selection risk for causal process reward models")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download-prm")
    p.add_argument("--output", required=True)
    p.add_argument("--files", nargs="+", default=["phase1_train.jsonl"])
    p.add_argument("--max-records", type=int)
    p = sub.add_parser("prepare-prm")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--blocked", nargs="+", required=True, help="Protected evaluation question JSONL files")
    p.add_argument("--neutral", choices=["mask", "positive"], default="mask")
    p.add_argument("--val-fraction", type=float, default=.1)
    p = sub.add_parser("budget-prm")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--step-budget", type=int, required=True)
    p.add_argument("--seed", type=int, default=2026)
    p = sub.add_parser("audit-splits")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("download-benchmark")
    p.add_argument("--name", choices=["math500", "gsm8k", "processbench"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--max-records", type=int)
    for name in ("train", "generate"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
    p = sub.add_parser("merge-caches")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("synthetic")
    p.add_argument("--output", required=True)
    p.add_argument("--train-questions", type=int, default=24)
    p.add_argument("--test-questions", type=int, default=12)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sizes", nargs="+", type=int, default=[2, 4])
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("evaluate-processbench")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.command == "download-prm":
        from .data import download_prm
        result = download_prm(args.output, args.files, max_records=args.max_records)
    elif args.command == "prepare-prm":
        from .data import prepare_prm
        result = prepare_prm(args.inputs, args.output, args.blocked, args.neutral, args.val_fraction)
    elif args.command == "budget-prm":
        from .data import budget_prm
        result = budget_prm(args.input, args.output, args.step_budget, args.seed)
    elif args.command == "audit-splits":
        from .data import audit_splits
        result = audit_splits(args.inputs)
        write_json(args.output, result)
        if not result["safe_exact"]: raise SystemExit("Exact cross-split overlap: see audit report")
    elif args.command == "download-benchmark":
        from .benchmarks import download_benchmark
        result = download_benchmark(args.name, args.output, args.revision, args.max_records)
    elif args.command == "train":
        from .train import train
        result = train(yaml.safe_load(Path(args.config).read_text()))
        result = {"output": result["config"]["output"], "cost": result["cost"]}
    elif args.command == "generate":
        from .generate import generate
        result = generate(yaml.safe_load(Path(args.config).read_text()))
    elif args.command == "merge-caches":
        from .io import sha256
        rows, seen, splits = [], set(), set()
        for path in args.inputs:
            for row in read_jsonl(path):
                key = (row["qid"], row["generator"], row["candidate_id"])
                if key in seen: raise ValueError("Overlapping cache identities")
                seen.add(key)
                splits.add(row["split"])
                rows.append(row)
        if len(splits) != 1: raise ValueError("Never merge different question splits")
        write_jsonl(args.output, rows)
        result = {"records": len(rows), "split": next(iter(splits)), "inputs": [{"path": p, "sha256": sha256(p)} for p in args.inputs]}
        write_json(Path(args.output).with_suffix(".manifest.json"), result)
    elif args.command == "synthetic":
        from .synthetic import make_synthetic
        result = make_synthetic(args.output, train_questions=args.train_questions, test_questions=args.test_questions)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        result = evaluate(args.checkpoint, args.candidates, args.output, args.sizes, args.device, args.bootstrap)
    else:
        from .benchmarks import evaluate_processbench
        result = evaluate_processbench(args.checkpoint, args.data, args.output, args.device, args.threshold)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
