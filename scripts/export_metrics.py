"""Export existing measured metrics; never fill missing formal experiments."""
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--comparison", default="results/tiny_comparison/comparison.json")
parser.add_argument("--output", default="results/tiny_comparison/tables")
args = parser.parse_args()
data = json.loads(Path(args.comparison).read_text())
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
rows = []
for method, result in data["methods"].items():
    for group, metrics in result["evaluation"]["groups"].items():
        rows.append({"method": method, "group": group, "questions": metrics["questions"],
                     "conditional_failure_lower": metrics["conditional_failure_bounds"][0],
                     "conditional_failure_upper": metrics["conditional_failure_bounds"][1],
                     "coverage_lower": metrics["coverage_bounds"][0], "coverage_upper": metrics["coverage_bounds"][1],
                     "accuracy_lower": metrics["overall_accuracy_bounds"][0], "accuracy_upper": metrics["overall_accuracy_bounds"][1]})
with (out / "metrics.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
tex = [r"% Synthetic arithmetic mechanism only; not PRM800K or real LLM results.",
       r"\begin{tabular}{llrrr}", r"Method & Group & Questions & Cond. failure bounds & Accuracy bounds \\", r"\hline"]
for row in rows:
    esc = lambda s: s.replace("_", r"\_")
    interval = lambda a, b: "--" if a is None else f"[{a:.3f}, {b:.3f}]"
    tex.append(f"{esc(row['method'])} & {esc(row['group'])} & {row['questions']} & "
               + interval(row["conditional_failure_lower"], row["conditional_failure_upper"]) + " & "
               + interval(row["accuracy_lower"], row["accuracy_upper"]) + r" \\")
tex.append(r"\end{tabular}")
(out / "metrics.tex").write_text("\n".join(tex) + "\n")
