"""Same candidate caches and step annotation file for all four methods.

Hyperparameters are explicit templates, not claimed tuned optima. A formal run
must freeze the validation-selected configuration before opening test metrics.
"""
from pathlib import Path
import argparse
import yaml
from prmshift.train import train
from prmshift.evaluate import evaluate
from prmshift.io import write_json

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/hf_1_5b.yaml")
parser.add_argument("--stage", choices=["train", "validation", "test"], required=True)
parser.add_argument("--output", default="results/formal")
parser.add_argument("--seed", type=int, default=2026)
args = parser.parse_args()
base = yaml.safe_load(Path(args.config).read_text())
reports = {}
for method in ("ce", "balanced_ce", "set_erm", "group_dro"):
    checkpoint = str(Path(args.output) / f"seed{args.seed}" / method / "checkpoint")
    if args.stage == "train":
        reports[method] = train(dict(base, seed=args.seed, method=method, output=checkpoint))["cost"]
    else:
        reports[method] = evaluate(checkpoint, f"data/candidates/{args.stage}_merged.jsonl",
            Path(args.output) / f"seed{args.seed}" / method / args.stage,
            sizes=[2, 4, 8] if args.stage == "validation" else [2, 4, 8, 16], device=base["device"])
write_json(Path(args.output) / f"seed{args.seed}" / f"{args.stage}_summary.json", reports)
