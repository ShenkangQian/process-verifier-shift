from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from .io import read_jsonl, write_json, write_jsonl, sha256
from .model import StepScorer, SequenceTooLong
from .objectives import build_pools
from .data import question_id


def pool_outcome(pool):
    candidates = pool["candidates"]
    ranked = sorted(candidates, key=lambda c: (-c["score"], str(c["candidate_id"])))
    selected = ranked[0]
    top2 = ranked[1]["score"] if len(ranked) > 1 else None
    known_correct = any(c["label"] is True for c in candidates)
    unknown = [c for c in candidates if c["label"] is None]
    selected_label = selected["label"]
    # An unknown selected answer can only be a conditional failure if another
    # answer could be correct. Bounds do not impute unknown as wrong.
    other_possible_correct = any(c["candidate_id"] != selected["candidate_id"] and c["label"] is not False for c in candidates)
    return {"qid": pool["qid"], "group": pool["group"], "generator": pool.get("generator", pool["group"].split("::")[0]),
            "selected_id": selected["candidate_id"],
            "top1_score": float(selected["score"]), "top2_score": None if top2 is None else float(top2),
            "margin": None if top2 is None else float(selected["score"] - top2), "top1_correct": selected_label,
            "coverage_lower": int(known_correct), "coverage_upper": int(known_correct or bool(unknown)),
            "failure_lower": int(known_correct and selected_label is False),
            "failure_upper": int(selected_label is not True and other_possible_correct),
            "accuracy_lower": int(selected_label is True), "accuracy_upper": int(selected_label is not False),
            "all_labels_known": not unknown, "unknown_candidates": len(unknown), "n": pool["n"]}


def aggregate(outcomes):
    n = len(outcomes)
    if not n: return {"questions": 0}
    totals = {key: sum(r[key] for r in outcomes) for key in
              ("coverage_lower", "coverage_upper", "failure_lower", "failure_upper", "accuracy_lower", "accuracy_upper")}
    cl, cu = totals["coverage_lower"], totals["coverage_upper"]
    fl, fu = totals["failure_lower"], totals["failure_upper"]
    all_known = all(r["all_labels_known"] for r in outcomes)
    lower = fl / cu if cu else None
    upper = min(1.0, fu / cl) if cl else (1.0 if fu else (0.0 if cu else None))
    al, au = totals["accuracy_lower"], totals["accuracy_upper"]
    # Proposition P2: accuracy = coverage x (1 - conditional failure), hence the
    # unconditional selection risk Pr(A|g) x R_g^hard equals coverage - accuracy.
    return {"questions": n, "all_labels_known": all_known,
            "unconditional_risk": fl / n if all_known else None, "unconditional_risk_bounds": [fl / n, fu / n],
            "error_rate": 1 - al / n if all_known else None, "error_rate_bounds": [1 - au / n, 1 - al / n],
            "coverage": cl / n if all_known else None, "coverage_bounds": [cl / n, cu / n],
            "conditional_failure": fl / cl if all_known and cl else None,
            "conditional_failure_bounds": [lower, upper],
            "overall_accuracy": totals["accuracy_lower"] / n if all_known else None,
            "overall_accuracy_bounds": [totals["accuracy_lower"] / n, totals["accuracy_upper"] / n],
            "unknown_candidate_fraction": sum(r["unknown_candidates"] for r in outcomes) / sum(r["n"] for r in outcomes),
            "known_covered_questions": cl, "possibly_covered_questions": cu}


BOUND_METRICS = ("coverage_bounds", "conditional_failure_bounds", "overall_accuracy_bounds",
                 "unconditional_risk_bounds", "error_rate_bounds")
RISK_CRITERIA = {"conditional": "conditional_failure_bounds", "unconditional": "unconditional_risk_bounds",
                 "error_rate": "error_rate_bounds"}


def worst_groups(group_reports):
    """Worst group under each risk criterion, from identification bounds.

    The midpoint ranking is the headline; a choice is 'robust' when the same
    group is worst under both the lower and the upper bound.
    """
    table = {}
    for criterion, metric in RISK_CRITERIA.items():
        rows = [(g, r[metric]) for g, r in group_reports.items() if r.get(metric) and r[metric][0] is not None]
        if not rows:
            table[criterion] = None
            continue
        by_mid = max(rows, key=lambda gr: ((gr[1][0] + gr[1][1]) / 2, gr[0]))[0]
        by_lower = max(rows, key=lambda gr: (gr[1][0], gr[0]))[0]
        by_upper = max(rows, key=lambda gr: (gr[1][1], gr[0]))[0]
        table[criterion] = {"group": by_mid, "by_lower_bound": by_lower, "by_upper_bound": by_upper,
                            "robust": by_lower == by_upper == by_mid,
                            "value_bounds": dict(rows)[by_mid]}
    conditional = table.get("conditional") and table["conditional"]["group"]
    unconditional = table.get("unconditional") and table["unconditional"]["group"]
    table["reversal"] = bool(conditional and unconditional and conditional != unconditional)
    return table


def summarize(outcomes, bootstrap=500, seed=2026):
    groups = defaultdict(list)
    for r in outcomes: groups[r["group"]].append(r)
    report = {g: aggregate(rows) for g, rows in sorted(groups.items())}
    rng = np.random.default_rng(seed)
    # Cluster resample question IDs globally. The same sampled question receives
    # the same multiplicity across generators and pool sizes.
    qids = sorted({r["qid"] for r in outcomes})
    samples = defaultdict(lambda: defaultdict(list))
    for _ in range(bootstrap if qids else 0):
        draw = rng.choice(qids, size=len(qids), replace=True)
        counts = {q: int(sum(draw == q)) for q in set(draw)}
        for g, rows in groups.items():
            sampled = [r for r in rows for _ in range(counts.get(r["qid"], 0))]
            summary = aggregate(sampled)
            for metric in BOUND_METRICS:
                if metric in summary and summary[metric][0] is not None:
                    samples[g][metric].append(summary[metric])
    for g in report:
        report[g]["question_bootstrap_95"] = {m: [float(np.quantile(v, .025, axis=0)[0]), float(np.quantile(v, .975, axis=0)[1])]
                                                 for m, v in samples[g].items() if v}
    by_generator = defaultdict(list)
    for r in outcomes: by_generator[r.get("generator", r["group"].split("::")[0])].append(r)
    generator_report = {g: aggregate(rows) for g, rows in sorted(by_generator.items())}
    return {"groups": report, "generators": generator_report,
            "worst_groups": {"by_group": worst_groups(report), "by_generator": worst_groups(generator_report)},
            "unique_questions": len(qids), "bootstrap_replicates": bootstrap,
            "bootstrap_unit": "question; shared multiplicities across generator × N",
            "bounds_note": "Conservative identification bounds for unknown answer labels, combined with percentile question bootstrap. Not an OOD guarantee."}


def evaluate(checkpoint, candidates_path, output, sizes=(2, 4), device="cpu", bootstrap=500, seed=2026, threads=2):
    torch.set_num_threads(threads)
    model = StepScorer.load(checkpoint, device).eval()
    candidates = list(read_jsonl(candidates_path))
    if any(c["qid"] != question_id(c["problem"]) for c in candidates):
        raise ValueError("Candidate question ID does not match canonical problem")
    if any(c.get("split") == "train" for c in candidates):
        raise ValueError("Held-out evaluation refuses training candidate caches")
    if len({c.get("split") for c in candidates}) > 1:
        raise ValueError("Evaluation cannot mix question splits in a single cache")
    training_report = Path(checkpoint) / "training_report.json"
    if training_report.exists():
        import json
        training = json.loads(training_report.read_text())
        train_ids = set(training["training_question_ids"])
        if train_ids & {c["qid"] for c in candidates}:
            raise ValueError("Question leakage: evaluation overlaps training steps")
    started = time.perf_counter()
    scored, invalid_qgroups = [], set()
    with torch.inference_mode():
        for c in candidates:
            try:
                scores = model.score_steps(c["problem"], c["steps"])
                scored.append(dict(c, score=float(scores.min()), step_logits=scores.tolist()))
            except SequenceTooLong:
                invalid_qgroups.add((c["qid"], c["generator"]))
    # Drop the entire original question-generator cache if any candidate exceeds
    # context. Never shorten N or selectively drop low/high scoring candidates.
    scored = [c for c in scored if (c["qid"], c["generator"]) not in invalid_qgroups]
    pools, pool_stats = build_pools(scored, sizes, seed, training=False)
    outcomes = [pool_outcome(p) for p in pools]
    report = summarize(outcomes, bootstrap, seed)
    report.update({"candidate_cache_sha256": sha256(candidates_path), "checkpoint": str(checkpoint),
                   "pool_statistics": pool_stats, "context_excluded_question_generator_pairs": len(invalid_qgroups),
                   "cost": {"wall_seconds": time.perf_counter() - started, "forward_input_tokens": model.token_count},
                   "scope": sorted({c.get("source", "unspecified") for c in candidates}),
                   "selection_tie_break": "lexicographic candidate_id", "pool_seed": seed})
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", report)
    write_jsonl(out / "scored_candidates.jsonl", scored)
    write_jsonl(out / "pool_outcomes.jsonl", outcomes)
    return report
