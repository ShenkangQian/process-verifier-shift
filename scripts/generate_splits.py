"""Generate immutable per-split caches and merge only within one split.

Not run on the development Mac. All generators are frozen. The held-out
generator never enters training or validation hyperparameter selection.
"""
from pathlib import Path
import argparse
import yaml
from prmshift.generate import generate
from prmshift.cli import main

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
args = parser.parse_args()
questions = {"train": "data/prepared/questions_train.jsonl", "validation": "data/prepared/questions_validation.jsonl",
             "test": "data/evaluation/math500.jsonl"}[args.split]
configs = ["configs/generate_qwen.yaml", "configs/generate_qwen_general.yaml"]
if args.split == "test": configs.append("configs/generate_heldout.yaml")
outputs = []
for path in configs:
    config = yaml.safe_load(Path(path).read_text())
    config["questions"] = questions
    config["output"] = f"data/candidates/{config['generator']}_{args.split}.jsonl"
    generate(config)
    outputs.append(config["output"])
main(["merge-caches", "--inputs", *outputs, "--output", f"data/candidates/{args.split}_merged.jsonl"])
