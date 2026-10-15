"""Reproducible four-baseline CPU mechanism study. No downloaded model."""
from pathlib import Path
import argparse
import json
import yaml
from prmshift.synthetic import make_synthetic
from prmshift.train import train
from prmshift.evaluate import evaluate
from prmshift.io import write_json

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=30)
parser.add_argument("--output", default="results/tiny_comparison")
args = parser.parse_args()
make_synthetic("data/synthetic", candidates=8)
config = yaml.safe_load(Path("configs/tiny_cpu.yaml").read_text())
config["iterations"] = args.iterations
reports = {}
for method in ("ce", "balanced_ce", "set_erm", "group_dro"):
    config = dict(config, method=method, output=str(Path(args.output) / method / "checkpoint"))
    training = train(config)
    evaluation = evaluate(config["output"], "data/synthetic/candidates_test.jsonl",
                          Path(args.output) / method / "evaluation", sizes=[2, 4, 8], bootstrap=200)
    reports[method] = {"cost": training["cost"], "initial_ce": training["history"][0]["step_ce"],
                       "last_ce": training["history"][-1]["step_ce"], "evaluation": evaluation}
    print(json.dumps({"method": method, "cost": training["cost"]}))
write_json(Path(args.output) / "comparison.json", {"scope": "synthetic_arithmetic_mechanism_only", "methods": reports})
