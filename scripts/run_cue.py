"""Spurious-cue mechanism study: P1 (soft vs hard), P2 (conditional vs unconditional), P3 (seen vs flipped cue).

Builds one cue dataset per seed, trains the four objectives on identical caches,
evaluates on test groups gen_A/gen_B/gen_C_heldout/gen_D_flipped x N in {2,4,8,16},
and aggregates across seeds with a (seed, question) cluster bootstrap.
Synthetic arithmetic only; nothing here is a PRM800K or real-LLM result.
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prmshift.io import read_jsonl, write_json  # noqa: E402

METHODS = ("ce", "balanced_ce", "set_erm", "group_dro")
TEST_SIZES = [2, 4, 8, 16]
SEEN_GENERATORS = ("gen_A", "gen_B", "gen_C_heldout")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--validate-every", type=int, default=20)
    parser.add_argument("--train-questions", type=int, default=300)
    parser.add_argument("--validation-questions", type=int, default=60)
    parser.add_argument("--test-questions", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--config", default="configs/cue_cpu.yaml")
    parser.add_argument("--data-root", default="data/synthetic_cue")
    parser.add_argument("--output", default="results/cue_comparison")
    parser.add_argument("--tau-ablation", type=float, nargs="*", default=[],
                        help="Extra temperatures for set_erm/group_dro, e.g. 0.1 0.3 (1.0 is the main run)")
    parser.add_argument("--tau-ablation-seeds", type=int, nargs="*", default=None,
                        help="Seeds for the ablation (default: first seed only)")
    parser.add_argument("--parallel", type=int, default=1, help="Seeds run concurrently as subprocesses")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose evaluation/metrics.json exists")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------- training


def run_seed(seed, args):
    import torch
    from prmshift.synthetic import make_synthetic_cue
    from prmshift.train import train
    from prmshift.evaluate import evaluate
    torch.set_num_threads(args.threads)
    data = ROOT / args.data_root / f"seed{seed}"
    out = ROOT / args.output / f"seed{seed}"
    started = time.perf_counter()
    manifest = make_synthetic_cue(data, seed, train_questions=args.train_questions,
                                  validation_questions=args.validation_questions, test_questions=args.test_questions)
    base = yaml.safe_load((ROOT / args.config).read_text())
    base.update({"seed": seed, "cpu_threads": args.threads, "iterations": args.iterations,
                 "validate_every": args.validate_every, "steps": str(data / "steps_train.jsonl"),
                 "candidates": str(data / "candidates_train.jsonl"),
                 "validation_steps": str(data / "steps_validation.jsonl"),
                 "validation_candidates": str(data / "candidates_validation.jsonl")})
    runs = [(m, base.get("temperature", 1.0), m) for m in args.methods]
    ablation_seeds = args.tau_ablation_seeds if args.tau_ablation_seeds is not None else [args.seeds[0]]
    if seed in ablation_seeds:
        for tau in args.tau_ablation:
            if abs(tau - base.get("temperature", 1.0)) < 1e-12:
                continue
            for m in ("set_erm", "group_dro"):
                if m in args.methods:
                    runs.append((m, tau, f"{m}_tau{tau:g}"))
    summary = {"seed": seed, "manifest": manifest, "runs": {}}
    for method, tau, name in runs:
        if args.skip_existing and (out / name / "evaluation/metrics.json").exists():
            print(json.dumps({"seed": seed, "run": name, "skipped": "existing"}), flush=True)
            continue
        config = dict(base, method=method, temperature=tau, output=str(out / name / "checkpoint"))
        t0 = time.perf_counter()
        training = train(config)
        t1 = time.perf_counter()
        evaluation = evaluate(config["output"], data / "candidates_test.jsonl", out / name / "evaluation",
                              sizes=TEST_SIZES, bootstrap=args.bootstrap, seed=seed, threads=args.threads)
        t2 = time.perf_counter()
        last_validation = training["validation_curve"][-1] if training["validation_curve"] else {}
        summary["runs"][name] = {"method": method, "temperature": tau, "train_seconds": t1 - t0,
                                 "evaluate_seconds": t2 - t1, "cost": training["cost"],
                                 "final_step_ce": training["history"][-1]["step_ce"],
                                 "final_validation_step_accuracy": last_validation.get("step_accuracy"),
                                 "final_validation_mean_soft": last_validation.get("mean_soft"),
                                 "final_validation_mean_hard": last_validation.get("mean_hard"),
                                 "worst_groups": evaluation["worst_groups"]}
        print(json.dumps({"seed": seed, "run": name, "train_s": round(t1 - t0, 1), "eval_s": round(t2 - t1, 1),
                          "val_step_acc": last_validation.get("step_accuracy")}), flush=True)
    summary["wall_seconds"] = time.perf_counter() - started
    previous = out / "summary.json"
    if previous.exists():
        # Merge with an earlier invocation (e.g. the temperature ablation added later).
        old_summary = json.loads(previous.read_text())
        summary["runs"] = {**old_summary.get("runs", {}), **summary["runs"]}
        summary["wall_seconds_previous_invocations"] = old_summary.get("wall_seconds_previous_invocations", 0) + old_summary.get("wall_seconds", 0)
    write_json(out / "summary.json", summary)
    return summary


# ----------------------------------------------------------------------------- aggregation


def cluster_bootstrap(outcomes_by_group, replicates, seed):
    """(seed, qid) cluster bootstrap over pooled outcomes; same multiplicity in every group."""
    from prmshift.evaluate import aggregate, BOUND_METRICS
    clusters = sorted({(r["seed"], r["qid"]) for rows in outcomes_by_group.values() for r in rows})
    index = {c: i for i, c in enumerate(clusters)}
    rng = np.random.default_rng(seed)
    samples = defaultdict(lambda: defaultdict(list))
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters))
        for g, rows in outcomes_by_group.items():
            sampled = [r for r in rows for _ in range(counts[index[(r["seed"], r["qid"])]])]
            summary = aggregate(sampled)
            for metric in BOUND_METRICS:
                if metric in summary and summary[metric][0] is not None:
                    samples[g][metric].append(summary[metric])
    return {g: {m: [float(np.quantile(v, .025, axis=0)[0]), float(np.quantile(v, .975, axis=0)[1])]
                for m, v in metrics.items() if v} for g, metrics in samples.items()}


def midpoint(bounds):
    return None if not bounds or bounds[0] is None else (bounds[0] + bounds[1]) / 2


def longest_plateau(values, tolerance=1e-9):
    """Longest run of consecutive checkpoints without a decrease."""
    best = run = 0
    for previous, current in zip(values, values[1:]):
        run = run + 1 if current >= previous - tolerance else 0
        best = max(best, run)
    return best


def curve_statistics(reports):
    """P1 diagnostics from validation curves, per N (mean over generators), per seed."""
    stats = {}
    per_n_curves = defaultdict(list)
    batch_curves = defaultdict(list)
    for report in reports:
        curve = report["validation_curve"]
        if not curve:
            continue
        iterations = [v["iteration"] for v in curve]
        by_n = defaultdict(lambda: {"soft": [], "hard": []})
        for v in curve:
            per_n = defaultdict(lambda: {"soft": [], "hard": []})
            for group, m in v["groups"].items():
                n = int(group.split("n=")[1])
                per_n[n]["soft"].append(m["soft"]); per_n[n]["hard"].append(m["hard"])
            for n, m in per_n.items():
                by_n[n]["soft"].append(float(np.mean(m["soft"]))); by_n[n]["hard"].append(float(np.mean(m["hard"])))
        for n, m in by_n.items():
            soft, hard = m["soft"], m["hard"]
            soft_decreases = sum(b < a - 1e-9 for a, b in zip(soft, soft[1:]))
            hard_flat_or_up_while_soft_down = sum(b < a - 1e-9 and d >= c - 1e-9
                                                  for a, b, c, d in zip(soft, soft[1:], hard, hard[1:]))
            entry = stats.setdefault(n, {"soft_monotone_fraction": [], "longest_hard_plateau": [],
                                         "soft_down_hard_not_down": [], "checkpoints": [],
                                         "soft_first_last": [], "hard_first_last": []})
            entry["soft_monotone_fraction"].append(soft_decreases / max(1, len(soft) - 1))
            entry["longest_hard_plateau"].append(longest_plateau(hard))
            entry["soft_down_hard_not_down"].append(hard_flat_or_up_while_soft_down)
            entry["checkpoints"].append(len(soft))
            entry["soft_first_last"].append([soft[0], soft[-1]])
            entry["hard_first_last"].append([hard[0], hard[-1]])
            per_n_curves[n].append((iterations, soft, hard))
        history = report["history"]
        by_n_batch = defaultdict(lambda: defaultdict(list))
        for h in history:
            # One value per iteration and N: mean over the generators sampled at that iteration.
            per_n = defaultdict(lambda: {"soft": [], "hard": []})
            for group, value in h.get("selection_soft_loss", {}).items():
                n = int(group.split("n=")[1])
                per_n[n]["soft"].append(value)
                per_n[n]["hard"].append(h["selection_hard_error"][group])
            for n, m in per_n.items():
                by_n_batch[n]["soft"].append((h["iteration"], float(np.mean(m["soft"]))))
                by_n_batch[n]["hard"].append((h["iteration"], float(np.mean(m["hard"]))))
        for n, m in by_n_batch.items():
            batch_curves[n].append(m)
    summary = {}
    for n, entry in sorted(stats.items()):
        summary[str(n)] = {k: (float(np.mean(v)) if k not in ("soft_first_last", "hard_first_last") else
                               [float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))])
                           for k, v in entry.items()}
        summary[str(n)]["seeds"] = len(entry["checkpoints"])
    return summary, per_n_curves, batch_curves


def aggregate_results(args):
    from prmshift.evaluate import aggregate, worst_groups, BOUND_METRICS
    out = ROOT / args.output
    seed_dirs = sorted(p for p in out.glob("seed*") if p.is_dir())
    runs = sorted({p.name for s in seed_dirs for p in s.iterdir() if p.is_dir() and (p / "evaluation/metrics.json").exists()})
    if not runs:
        raise SystemExit("No completed runs to aggregate")
    result = {"scope": "synthetic_arithmetic_cue_mechanism_only", "seeds": [], "runs": {}, "curves": {},
              "bootstrap": {"replicates": args.bootstrap, "unit": "(seed, question) cluster; shared multiplicity across groups"}}
    curves_for_figures = {}
    margins = {}
    per_seed_summaries = {}
    for sd in seed_dirs:
        summary_path = sd / "summary.json"
        if summary_path.exists():
            per_seed_summaries[sd.name] = json.loads(summary_path.read_text())
            result["seeds"].append(int(sd.name[4:]))
    for run in runs:
        outcomes_by_group, outcomes_by_generator = defaultdict(list), defaultdict(list)
        per_seed_groups = defaultdict(dict)
        reports, metrics_by_seed = [], {}
        run_margins = {"correct": [], "incorrect": []}
        for sd in seed_dirs:
            seed = int(sd.name[4:])
            metrics_path = sd / run / "evaluation/metrics.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text())
            metrics_by_seed[seed] = metrics
            for group, m in metrics["groups"].items():
                per_seed_groups[group][seed] = m
            for r in read_jsonl(sd / run / "evaluation/pool_outcomes.jsonl"):
                r["seed"] = seed
                outcomes_by_group[r["group"]].append(r)
                outcomes_by_generator[r["generator"]].append(r)
                if r["margin"] is not None and r["top1_correct"] is not None:
                    run_margins["correct" if r["top1_correct"] else "incorrect"].append(r["margin"])
            reports.append(json.loads((sd / run / "checkpoint/training_report.json").read_text()))
        pooled_groups = {g: aggregate(rows) for g, rows in sorted(outcomes_by_group.items())}
        pooled_generators = {g: aggregate(rows) for g, rows in sorted(outcomes_by_generator.items())}
        ci_groups = cluster_bootstrap(outcomes_by_group, args.bootstrap, 2026)
        ci_generators = cluster_bootstrap(outcomes_by_generator, args.bootstrap, 2026)
        for g in pooled_groups:
            pooled_groups[g]["cluster_bootstrap_95"] = ci_groups.get(g, {})
            seeds_mid = [midpoint(m["conditional_failure_bounds"]) for m in per_seed_groups[g].values()]
            seeds_mid = [x for x in seeds_mid if x is not None]
            pooled_groups[g]["per_seed_conditional_failure_midpoint"] = {
                "values": {str(s): midpoint(m["conditional_failure_bounds"]) for s, m in sorted(per_seed_groups[g].items())},
                "mean": float(np.mean(seeds_mid)) if seeds_mid else None,
                "std": float(np.std(seeds_mid, ddof=1)) if len(seeds_mid) > 1 else None}
        for g in pooled_generators:
            pooled_generators[g]["cluster_bootstrap_95"] = ci_generators.get(g, {})
        per_seed_worst = {str(s): m["worst_groups"] for s, m in sorted(metrics_by_seed.items())}
        curve_summary, per_n_curves, batch_curves = curve_statistics(reports)
        curves_for_figures[run] = (per_n_curves, batch_curves)
        margins[run] = run_margins
        training = {"seeds": len(reports),
                    "final_validation_step_accuracy": [r["validation_curve"][-1]["step_accuracy"] for r in reports if r["validation_curve"]],
                    "initial_validation_step_accuracy": [r["validation_curve"][0]["step_accuracy"] for r in reports if r["validation_curve"]],
                    "final_step_ce": [r["history"][-1]["step_ce"] for r in reports],
                    "train_wall_seconds": [r["cost"]["wall_seconds"] for r in reports],
                    "forward_input_tokens": [r["cost"]["forward_input_tokens"] for r in reports],
                    "temperature": reports[0]["config"].get("temperature") if reports else None,
                    "iterations": reports[0]["config"].get("iterations") if reports else None}
        result["runs"][run] = {"method": reports[0]["config"]["method"] if reports else run,
                               "groups": pooled_groups, "generators": pooled_generators,
                               "worst_groups_pooled": {"by_group": worst_groups(pooled_groups),
                                                       "by_generator": worst_groups(pooled_generators)},
                               "worst_groups_per_seed": per_seed_worst,
                               "p1_validation_curve_statistics": curve_summary,
                               "margins": {k: {"count": len(v), "mean": float(np.mean(v)) if v else None,
                                               "median": float(np.median(v)) if v else None} for k, v in run_margins.items()},
                               "training": training}
    result["scorecard_numbers"] = scorecard_numbers(result)
    result["per_seed_wall_seconds"] = {k: v.get("wall_seconds") for k, v in per_seed_summaries.items()}
    write_json(out / "aggregate.json", result)
    write_tables(result, out / "tables")
    make_figures(result, curves_for_figures, margins, out / "figures")
    return result


def scorecard_numbers(result):
    """Numbers the PREDICTIONS.md scorecard is judged on. Verdicts are written by hand from these."""
    runs = result["runs"]
    numbers = {}
    main = {name: r for name, r in runs.items() if name in METHODS}

    def worst_conditional(groups, generators):
        rows = [(g, m) for g, m in groups.items() if g.split("::")[0] in generators
                and m["conditional_failure_bounds"][0] is not None]
        if not rows:
            return None
        g, m = max(rows, key=lambda gm: midpoint(gm[1]["conditional_failure_bounds"]))
        return {"group": g, "midpoint": midpoint(m["conditional_failure_bounds"]),
                "bounds": m["conditional_failure_bounds"],
                "ci95": m["cluster_bootstrap_95"].get("conditional_failure_bounds")}
    numbers["H1_worst_group_conditional_failure_seen_and_heldout"] = {
        name: worst_conditional(r["groups"], SEEN_GENERATORS) for name, r in main.items()}
    numbers["H1_per_generator_conditional_failure"] = {
        name: {g: {"midpoint": midpoint(r["generators"][g]["conditional_failure_bounds"]),
                   "ci95": r["generators"][g]["cluster_bootstrap_95"].get("conditional_failure_bounds")}
               for g in SEEN_GENERATORS if g in r["generators"]} for name, r in main.items()}
    numbers["H2_gen_D_flipped_conditional_failure"] = {
        name: {g: {"midpoint": midpoint(m["conditional_failure_bounds"]),
                   "ci95": m["cluster_bootstrap_95"].get("conditional_failure_bounds")}
               for g, m in r["groups"].items() if g.startswith("gen_D_flipped")}
        | {"generator_pooled": {"midpoint": midpoint(r["generators"]["gen_D_flipped"]["conditional_failure_bounds"]),
                                "ci95": r["generators"]["gen_D_flipped"]["cluster_bootstrap_95"].get("conditional_failure_bounds")}
           if "gen_D_flipped" in r["generators"] else None}
        for name, r in main.items()}
    numbers["H3_worst_group_reversal"] = {
        name: {"pooled_by_group": {k: (v["group"] if isinstance(v, dict) else v)
                                   for k, v in r["worst_groups_pooled"]["by_group"].items()},
               "pooled_by_generator": {k: (v["group"] if isinstance(v, dict) else v)
                                       for k, v in r["worst_groups_pooled"]["by_generator"].items()},
               "per_seed_reversal_by_group": {s: w["by_group"]["reversal"] for s, w in r["worst_groups_per_seed"].items()},
               "per_seed_unconditional_worst_by_group": {s: w["by_group"]["unconditional"]["group"] for s, w in r["worst_groups_per_seed"].items()},
               "per_seed_conditional_worst_by_group": {s: w["by_group"]["conditional"]["group"] for s, w in r["worst_groups_per_seed"].items()}}
        for name, r in main.items()}
    numbers["H4_validation_curves"] = {name: r["p1_validation_curve_statistics"] for name, r in main.items()}
    numbers["step_accuracy_check"] = {name: r["training"]["final_validation_step_accuracy"] for name, r in main.items()}
    return numbers


def write_tables(result, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run, r in result["runs"].items():
        for group, m in r["groups"].items():
            ci = m["cluster_bootstrap_95"]
            rows.append({"run": run, "method": r["method"], "temperature": r["training"]["temperature"], "group": group,
                         "pools": m["questions"], "seeds": r["training"]["seeds"],
                         "conditional_failure_lower": m["conditional_failure_bounds"][0],
                         "conditional_failure_upper": m["conditional_failure_bounds"][1],
                         "conditional_failure_ci95_low": (ci.get("conditional_failure_bounds") or [None, None])[0],
                         "conditional_failure_ci95_high": (ci.get("conditional_failure_bounds") or [None, None])[1],
                         "per_seed_conditional_failure_mean": m["per_seed_conditional_failure_midpoint"]["mean"],
                         "per_seed_conditional_failure_std": m["per_seed_conditional_failure_midpoint"]["std"],
                         "coverage_lower": m["coverage_bounds"][0], "coverage_upper": m["coverage_bounds"][1],
                         "unconditional_risk_lower": m["unconditional_risk_bounds"][0],
                         "unconditional_risk_upper": m["unconditional_risk_bounds"][1],
                         "accuracy_lower": m["overall_accuracy_bounds"][0], "accuracy_upper": m["overall_accuracy_bounds"][1],
                         "accuracy_ci95_low": (ci.get("overall_accuracy_bounds") or [None, None])[0],
                         "accuracy_ci95_high": (ci.get("overall_accuracy_bounds") or [None, None])[1]})
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    esc = lambda s: str(s).replace("_", r"\_")
    interval = lambda a, b: "--" if a is None else f"[{a:.3f}, {b:.3f}]"
    tex = [r"% Synthetic arithmetic cue mechanism only; not PRM800K or real LLM results.",
           r"% Bounds: identification bounds pooled over seeds; CI: (seed, question) cluster bootstrap 95%.",
           r"\begin{tabular}{llrlll}", r"Run & Group & Pools & Cond. failure bounds & Cond. failure 95\% CI & Accuracy bounds \\", r"\hline"]
    for row in rows:
        tex.append(f"{esc(row['run'])} & {esc(row['group'])} & {row['pools']} & "
                   + interval(row["conditional_failure_lower"], row["conditional_failure_upper"]) + " & "
                   + interval(row["conditional_failure_ci95_low"], row["conditional_failure_ci95_high"]) + " & "
                   + interval(row["accuracy_lower"], row["accuracy_upper"]) + r" \\")
    tex.append(r"\end{tabular}")
    (out / "metrics.tex").write_text("\n".join(tex) + "\n")
    (out / "worst_groups.txt").write_text(worst_group_text(result))
    write_json(out / "scorecard_numbers.json", result["scorecard_numbers"])


def worst_group_text(result):
    lines = ["Worst group per method (pooled over seeds; midpoint of identification bounds; 'robust' = same under lower and upper bound)", ""]
    for level in ("by_group", "by_generator"):
        lines.append(f"[{level}]")
        lines.append(f"{'run':<16} {'conditional (cf)':<28} {'unconditional (cov x cf)':<28} {'error rate (1-acc)':<28} reversal")
        for run, r in result["runs"].items():
            w = r["worst_groups_pooled"][level]
            cells = []
            for criterion in ("conditional", "unconditional", "error_rate"):
                entry = w[criterion]
                if entry is None:
                    cells.append("--")
                else:
                    lo, hi = entry["value_bounds"]
                    cells.append(f"{entry['group']} [{lo:.3f},{hi:.3f}]{'' if entry['robust'] else ' (not robust)'}")
            lines.append(f"{run:<16} {cells[0]:<28} {cells[1]:<28} {cells[2]:<28} {'yes' if w['reversal'] else 'no'}")
        lines.append("")
    lines.append("Per-seed reversal (conditional worst != unconditional worst, by_group):")
    for run, r in result["runs"].items():
        flags = {s: w["by_group"]["reversal"] for s, w in r["worst_groups_per_seed"].items()}
        lines.append(f"  {run:<16} " + ", ".join(f"seed{s}: {'yes' if f else 'no'}" for s, f in flags.items()))
    return "\n".join(lines) + "\n"


def make_figures(result, curves, margins, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    main_runs = [m for m in METHODS if m in result["runs"]]
    colors = {"ce": "#4C72B0", "balanced_ce": "#8172B2", "set_erm": "#DD8452", "group_dro": "#55A868"}
    generators = ["gen_A", "gen_B", "gen_C_heldout", "gen_D_flipped"]
    # (a) conditional failure per method per generator, faceted by N.
    fig, axes = plt.subplots(1, len(TEST_SIZES), figsize=(4.2 * len(TEST_SIZES), 4), sharey=True)
    width = 0.8 / max(1, len(main_runs))
    for ax, n in zip(axes, TEST_SIZES):
        for mi, run in enumerate(main_runs):
            groups = result["runs"][run]["groups"]
            xs, ys, lo, hi = [], [], [], []
            for gi, g in enumerate(generators):
                m = groups.get(f"{g}::n={n}")
                if not m or m["conditional_failure_bounds"][0] is None:
                    continue
                mid = midpoint(m["conditional_failure_bounds"])
                ci = m["cluster_bootstrap_95"].get("conditional_failure_bounds", m["conditional_failure_bounds"])
                xs.append(gi + (mi - (len(main_runs) - 1) / 2) * width); ys.append(mid)
                lo.append(max(0.0, mid - ci[0])); hi.append(max(0.0, ci[1] - mid))
            ax.bar(xs, ys, width=width, color=colors.get(run, "gray"), label=run, yerr=[lo, hi], capsize=2, error_kw={"lw": 0.8})
        ax.set_title(f"N = {n}")
        ax.set_xticks(range(len(generators)))
        ax.set_xticklabels([g.replace("_heldout", "\n(heldout)").replace("_flipped", "\n(flipped)") for g in generators], fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("conditional failure (midpoint of bounds)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Conditional selection failure on test pools, pooled over {len(result['seeds'])} seeds; error bars: (seed, question) cluster bootstrap 95%", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_a_conditional_failure_by_generator_and_n.png", dpi=150)
    plt.close(fig)
    # (b) soft vs hard training curves per method.
    fig, axes = plt.subplots(2, len(main_runs), figsize=(4.2 * len(main_runs), 6.5), sharex=True, squeeze=False)
    n_colors = {2: "#1b9e77", 4: "#d95f02", 8: "#7570b3", 16: "#e7298a"}
    for col, run in enumerate(main_runs):
        per_n_curves, batch_curves = curves[run]
        for n in sorted(per_n_curves):
            entries = per_n_curves[n]
            length = min(len(e[0]) for e in entries)
            iterations = entries[0][0][:length]
            soft = np.mean([e[1][:length] for e in entries], axis=0)
            hard = np.mean([e[2][:length] for e in entries], axis=0)
            axes[0][col].plot(iterations, soft, color=n_colors.get(n, "k"), label=f"N={n} validation")
            axes[1][col].plot(iterations, hard, color=n_colors.get(n, "k"), label=f"N={n} validation")
            if n in batch_curves:
                window = 50
                for row, key in ((0, "soft"), (1, "hard")):
                    series = [np.array([v for _, v in seed_curve[key]]) for seed_curve in batch_curves[n]]
                    length_b = min(len(s) for s in series)
                    mean = np.mean([s[:length_b] for s in series], axis=0)
                    if len(mean) >= window:
                        smooth = np.convolve(mean, np.ones(window) / window, mode="valid")
                        axes[row][col].plot(np.arange(window, len(mean) + 1), smooth, color=n_colors.get(n, "k"),
                                            alpha=0.35, lw=0.8, ls="--")
        axes[0][col].set_title(run)
        axes[1][col].set_xlabel("iteration")
        for row in (0, 1):
            axes[row][col].grid(alpha=0.3)
    axes[0][0].set_ylabel("soft selection loss (tau = training)")
    axes[1][0].set_ylabel("hard selection error")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Soft vs hard selection on validation pools (solid, mean over generators and seeds); dashed: sampled-batch running mean (window 50)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_b_soft_vs_hard_training_curves.png", dpi=150)
    plt.close(fig)
    # (c) margin histograms split by top-1 correctness.
    fig, axes = plt.subplots(1, len(main_runs), figsize=(4.2 * len(main_runs), 3.8), sharey=True, squeeze=False)
    for ax, run in zip(axes[0], main_runs):
        m = margins[run]
        all_values = m["correct"] + m["incorrect"]
        if all_values:
            bins = np.linspace(0, max(all_values) + 1e-9, 40)
            ax.hist(m["correct"], bins=bins, alpha=0.6, color="#55A868", label=f"top-1 correct (n={len(m['correct'])})")
            ax.hist(m["incorrect"], bins=bins, alpha=0.6, color="#C44E52", label=f"top-1 incorrect (n={len(m['incorrect'])})")
        ax.set_title(run)
        ax.set_xlabel("top-1 minus top-2 score")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("test pools (all groups, all seeds)")
    fig.suptitle("Selection margins on test pools; unknown-label selections excluded", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_c_margin_histograms.png", dpi=150)
    plt.close(fig)
    # (d) worst-group table as text.
    text = worst_group_text(result)
    fig = plt.figure(figsize=(14, 0.22 * (text.count("\n") + 2) + 0.5))
    fig.text(0.01, 0.99, text, family="monospace", fontsize=7.5, va="top")
    fig.savefig(out / "fig_d_worst_group_table.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        for seed in args.seeds:
            run_seed(seed, args)
        return
    started = time.perf_counter()
    if not args.aggregate_only:
        if args.parallel > 1 and len(args.seeds) > 1:
            pending = list(args.seeds)
            running = []
            child_args = [a for a in (sys.argv[1:] if argv is None else argv) if a not in ("--parallel",)]
            # Drop the value that followed --parallel.
            filtered, skip = [], False
            for a in (sys.argv[1:] if argv is None else argv):
                if skip: skip = False; continue
                if a == "--parallel": skip = True; continue
                if a.startswith("--seeds"): continue
                filtered.append(a)
            filtered = [a for a in filtered if a not in map(str, args.seeds)]
            while pending or running:
                while pending and len(running) < args.parallel:
                    seed = pending.pop(0)
                    running.append(subprocess.Popen([sys.executable, __file__, "--worker", "--seeds", str(seed), *filtered], cwd=ROOT))
                for proc in list(running):
                    if proc.poll() is not None:
                        running.remove(proc)
                        if proc.returncode != 0:
                            raise SystemExit(f"Worker failed with exit code {proc.returncode}")
                time.sleep(2)
        else:
            for seed in args.seeds:
                run_seed(seed, args)
    result = aggregate_results(args)
    result["total_wall_seconds_this_invocation"] = time.perf_counter() - started
    write_json(ROOT / args.output / "aggregate.json", result)
    print(json.dumps({"seeds": result["seeds"], "runs": list(result["runs"]),
                      "wall_seconds": round(result["total_wall_seconds_this_invocation"], 1)}))


if __name__ == "__main__":
    main()
