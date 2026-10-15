from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import random
import time
import torch
import torch.nn.functional as F
from .io import read_jsonl, write_json, sha256
from .model import StepScorer, causal_text, SequenceTooLong
from .objectives import build_pools, soft_selection_loss, hard_selection_error, GroupDRO
from .data import question_id


def _load_validation(config, model, train_qids, seed):
    """Validation pools (unknown labels removed, all-wrong excluded) and labelled steps."""
    pools_by_group, steps = {}, []
    if config.get("validation_candidates"):
        rows = list(read_jsonl(config["validation_candidates"]))
        if any(c.get("split") == "train" or c["qid"] in train_qids for c in rows):
            raise ValueError("Validation candidate cache overlaps training questions")
        pools, _ = build_pools(rows, config.get("pool_sizes", [2, 4]), seed, training=True)
        for pool in pools:
            try:
                for candidate in pool["candidates"]:
                    model.encode([causal_text(candidate["problem"], candidate["steps"][:-1], candidate["steps"][-1])])
            except SequenceTooLong:
                continue
            pools_by_group.setdefault(pool["group"], []).append(pool)
    if config.get("validation_steps"):
        for row in read_jsonl(config["validation_steps"]):
            if row.get("split") == "train" or row["qid"] in train_qids:
                raise ValueError("Validation steps overlap training questions")
            if row.get("label") in (0, 1):
                try:
                    model.encode([causal_text(row["problem"], row["prefix"], row["step"])])
                    steps.append(row)
                except SequenceTooLong:
                    continue
    return pools_by_group, steps


def _validate(model, pools_by_group, steps, temperature, device, batch=32):
    """Soft selection loss and hard selection error on every validation pool, plus step accuracy."""
    result = {"groups": {}, "step_accuracy": None, "step_bce": None, "labelled_steps": len(steps)}
    with torch.no_grad():
        unique = {}
        for pools in pools_by_group.values():
            for pool in pools:
                for c in pool["candidates"]:
                    unique.setdefault((c["qid"], c["generator"], c["candidate_id"]), c)
        keys = list(unique)
        cache = dict(zip(keys, model.score_candidates([unique[k] for k in keys]))) if keys else {}
        for group, pools in pools_by_group.items():
            soft, hard = [], []
            for pool in pools:
                scores = torch.stack([cache[(c["qid"], c["generator"], c["candidate_id"])] for c in pool["candidates"]])
                labels = [c["label"] for c in pool["candidates"]]
                soft.append(float(soft_selection_loss(scores, labels, temperature)))
                hard.append(hard_selection_error(scores, labels))
            result["groups"][group] = {"soft": sum(soft) / len(soft), "hard": sum(hard) / len(hard), "pools": len(pools)}
        if steps:
            correct, bce = 0, 0.0
            for i in range(0, len(steps), batch):
                chunk = steps[i:i + batch]
                logits = model.score_texts([causal_text(r["problem"], r["prefix"], r["step"]) for r in chunk])
                targets = torch.tensor([r["label"] for r in chunk], device=device, dtype=logits.dtype)
                correct += int(((logits > 0).to(targets.dtype) == targets).sum())
                bce += float(F.binary_cross_entropy_with_logits(logits, targets, reduction="sum"))
            result["step_accuracy"] = correct / len(steps)
            result["step_bce"] = bce / len(steps)
    if result["groups"]:
        result["mean_soft"] = sum(v["soft"] for v in result["groups"].values()) / len(result["groups"])
        result["mean_hard"] = sum(v["hard"] for v in result["groups"].values()) / len(result["groups"])
    return result


def train(config):
    seed = int(config.get("seed", 2026))
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("cpu_threads", 2))
    rng = random.Random(seed)
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This configuration requires NVIDIA CUDA; use tiny_cpu.yaml locally")
    method = config.get("method", "group_dro")
    if method not in ("ce", "balanced_ce", "set_erm", "group_dro"):
        raise ValueError("Unsupported baseline; PRISM is an external, unimplemented baseline")
    rows = list(read_jsonl(config["steps"]))
    if any(r["qid"] != question_id(r["problem"]) for r in rows):
        raise ValueError("Step question ID does not match canonical problem")
    if any(r.get("split") not in (None, "train") for r in rows):
        raise ValueError("Training step file contains a non-training question")
    rows = [r for r in rows if r.get("label") in (0, 1)]
    if not rows: raise ValueError("No labelled training steps")
    resume = config.get("resume")
    model = StepScorer.load(resume, device) if resume else StepScorer(config["model"]).to(device)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=config.get("learning_rate", 0.001), weight_decay=config.get("weight_decay", 0.01))
    candidates = list(read_jsonl(config["candidates"])) if config.get("candidates") else []
    if any(c["qid"] != question_id(c["problem"]) for c in candidates):
        raise ValueError("Candidate question ID does not match canonical problem")
    if any(c.get("split") not in (None, "train") for c in candidates):
        raise ValueError("Candidate cache contains non-training questions")
    # A train cache may cover a subset of step-supervised questions; it may not
    # silently introduce validation/test questions through a mislabeled cache.
    qids = {r["qid"] for r in rows}
    if candidates and any(c["qid"] not in qids for c in candidates):
        raise ValueError("Candidate question is absent from the training step split")
    pools, pool_stats = build_pools(candidates, config.get("pool_sizes", [2, 4]), seed, training=True)
    groups = defaultdict(list)
    for pool in pools: groups[pool["group"]].append(pool)
    if method in ("set_erm", "group_dro") and not groups:
        raise ValueError("No eligible conditional-selection groups")
    expected_groups = {f"{c['generator']}::n={n}" for c in candidates for n in config.get("pool_sizes", [2, 4])}
    missing_groups = sorted(expected_groups - set(groups))
    if method in ("set_erm", "group_dro") and missing_groups and config.get("require_all_groups", True):
        raise ValueError(f"Requested groups have no eligible pools: {missing_groups}. Inspect unknown labels and candidate counts.")
    dro = GroupDRO(groups, config.get("dro_eta", 0.1))
    positive = sum(r["label"] == 1 for r in rows)
    negative = len(rows) - positive
    if method == "balanced_ce" and (positive == 0 or negative == 0):
        raise ValueError("Balanced CE needs both step classes")
    class_weights = {0: len(rows) / max(1, 2 * negative), 1: len(rows) / max(1, 2 * positive)}
    history = []
    completed = 0
    start_step = 0
    signature = {"method": method, "model": config["model"], "seed": seed,
        "pool_sizes": config.get("pool_sizes", [2, 4]), "temperature": config.get("temperature", 1.0),
        "ce_weight": config.get("ce_weight", 1.0), "dro_eta": config.get("dro_eta", .1),
        "step_batch_size": config.get("step_batch_size", 8), "gradient_clip": config.get("gradient_clip", 1.),
        "learning_rate": config.get("learning_rate", .001), "weight_decay": config.get("weight_decay", .01),
        "steps_sha256": sha256(config["steps"]), "candidates_sha256": sha256(config["candidates"]) if candidates else None}
    if resume:
        # Checkpoint is produced by this repository; torch weights_only still
        # permits optimizer tensors and primitive RNG state, never arbitrary code.
        state = torch.load(Path(resume) / "training_state.pt", weights_only=True, map_location=device)
        if state["method"] != method: raise ValueError("Resume method differs")
        if state["signature"] != signature: raise ValueError("Resume data hash or objective/model/sampling configuration differs")
        opt.load_state_dict(state["optimizer"])
        dro.load_state_dict(state["dro"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step = state["step"]
        completed = start_step
    start = time.perf_counter()
    # Explicitly reject overlong prefixes ahead of training. No selective
    # dropping after looking at labels or current model loss.
    overlong_steps = 0
    valid_rows = []
    for row in rows:
        try:
            model.encode([causal_text(row["problem"], row["prefix"], row["step"])])
            valid_rows.append(row)
        except SequenceTooLong:
            overlong_steps += 1
    rows = valid_rows
    if not rows: raise ValueError("All step prefixes exceed the context limit")
    positive = sum(r["label"] == 1 for r in rows)
    negative = len(rows) - positive
    if method == "balanced_ce" and (positive == 0 or negative == 0):
        raise ValueError("Context-compatible steps must contain both classes for balanced CE")
    class_weights = {0: len(rows) / max(1, 2 * negative), 1: len(rows) / max(1, 2 * positive)}
    for group in list(groups):
        valid_pools = []
        for pool in groups[group]:
            try:
                for candidate in pool["candidates"]:
                    model.encode([causal_text(candidate["problem"], candidate["steps"][:-1], candidate["steps"][-1])])
                valid_pools.append(pool)
            except SequenceTooLong:
                pool_stats["overlong_pools_excluded"] = pool_stats.get("overlong_pools_excluded", 0) + 1
        if not valid_pools and method in ("set_erm", "group_dro"):
            raise ValueError(f"No context-compatible pools in group {group}; change length configuration")
        groups[group] = valid_pools
    validation_pools, validation_steps = _load_validation(config, model, qids, seed)
    validate_every = int(config.get("validate_every", 10))
    temperature = config.get("temperature", 1.0)
    log_diagnostics = config.get("log_selection_diagnostics", True)
    validation_curve = []
    diagnostic_tokens = 0
    model.token_count = 0

    def run_validation(iteration):
        nonlocal diagnostic_tokens
        if not validation_pools and not validation_steps: return
        before = model.token_count
        record = _validate(model, validation_pools, validation_steps, temperature, device)
        diagnostic_tokens += model.token_count - before
        model.token_count = before
        record["iteration"] = iteration
        validation_curve.append(record)

    if not resume: run_validation(start_step)
    for iteration in range(start_step, config.get("iterations", 100)):
        opt.zero_grad(set_to_none=True)
        batch = rng.choices(rows, k=config.get("step_batch_size", 8))
        logits = model.score_texts([causal_text(r["problem"], r["prefix"], r["step"]) for r in batch])
        targets = torch.tensor([r["label"] for r in batch], device=device, dtype=logits.dtype)
        point_losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if method == "balanced_ce":
            weights = torch.tensor([class_weights[r["label"]] for r in batch], device=device, dtype=logits.dtype)
            ce = (point_losses * weights).mean()
        else:
            ce = point_losses.mean()
        anchor = config.get("ce_weight", 1.0) if method in ("set_erm", "group_dro") else 1.0
        (anchor * ce).backward()
        group_losses, batch_soft, batch_hard = {}, {}, {}
        # Accumulate separate group graphs instead of retaining every group's
        # activations. Group sampling is uniform over eligible questions.
        weights = dro.weights(device) if method == "group_dro" else None
        if method in ("set_erm", "group_dro"):
            for gi, group in enumerate(dro.groups):
                pool = rng.choice(groups[group])
                coefficient = weights[gi] if weights is not None else 1.0 / len(groups)
                loss, scores = model.selection_backward(pool["candidates"], temperature, coefficient, return_scores=True)
                group_losses[group] = loss.detach()
                batch_soft[group] = float(loss.detach())
                batch_hard[group] = hard_selection_error(scores, [c["label"] for c in pool["candidates"]])
        elif log_diagnostics and groups:
            # CE baselines: same sampled-pool soft/hard diagnostics, no gradient,
            # not counted as training forward tokens.
            before = model.token_count
            with torch.no_grad():
                for group in dro.groups:
                    pool = rng.choice(groups[group])
                    scores = model.score_candidates(pool["candidates"])
                    labels = [c["label"] for c in pool["candidates"]]
                    batch_soft[group] = float(soft_selection_loss(scores, labels, temperature))
                    batch_hard[group] = hard_selection_error(scores, labels)
            diagnostic_tokens += model.token_count - before
            model.token_count = before
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.0)))
        if not torch.isfinite(torch.tensor(grad_norm)): raise FloatingPointError("Non-finite gradient")
        opt.step()
        if method == "group_dro": dro.update(group_losses)
        completed = iteration + 1
        history.append({"iteration": completed, "step_ce": float(ce.detach()), "gradient_norm": grad_norm,
                        "selection_risk": {g: float(v) for g, v in group_losses.items()},
                        "selection_soft_loss": batch_soft, "selection_hard_error": batch_hard,
                        "dro_weights": dict(zip(dro.groups, dro.weights().tolist())),
                        "forward_input_tokens": model.token_count})
        if validate_every > 0 and completed % validate_every == 0:
            run_validation(completed)
        # Check only at a completed optimizer step; report actual overshoot.
        if config.get("max_forward_tokens") and model.token_count >= config["max_forward_tokens"]:
            break
    out = Path(config["output"])
    model.save(out)
    torch.save({"optimizer": opt.state_dict(), "dro": dro.state_dict(), "method": method, "signature": signature,
                "step": completed, "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None}, out / "training_state.pt")
    cost = {"wall_seconds": time.perf_counter() - start, "optimizer_steps_this_run": completed - start_step,
            "forward_input_tokens": model.token_count, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()), "device": device,
            "measurement": "Actual prefix forward tokens, including repeated prefixes; excludes preprocessing and generation. Not FLOPs.",
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None}
    if validation_curve and validation_curve[-1]["iteration"] != completed:
        run_validation(completed)
    cost["diagnostic_forward_tokens"] = diagnostic_tokens
    cost["diagnostic_note"] = "Sampled-pool soft/hard diagnostics and validation passes; excluded from forward_input_tokens."
    result = {"config": config, "history": history, "cost": cost, "pool_statistics": pool_stats,
              "validation_curve": validation_curve, "validate_every": validate_every,
              "validation_groups": {g: len(ps) for g, ps in validation_pools.items()},
              "selection_metrics_note": "selection_soft_loss is the softmax selection loss at the training temperature on the one pool sampled per group at that iteration; selection_hard_error is 1 when the argmax candidate of that pool is incorrect. Validation entries average over all eligible validation pools.",
              "masked_or_unlabelled_steps": sum(r.get("label") is None for r in read_jsonl(config["steps"])),
              "overlong_steps_excluded": overlong_steps,
              "data_sha256": {"steps": sha256(config["steps"]), "candidates": sha256(config["candidates"]) if candidates else None},
              "training_question_ids": sorted(qids),
              "observed_conditional_groups": {g: len(ps) for g, ps in groups.items()},
              "requested_groups_without_eligible_pools": missing_groups,
              "result_scope": "synthetic_mechanism" if config["model"].get("kind") == "tiny" else "configured_data_run",
              "external_baselines": {"PRISM": "not_implemented_not_evaluated"}}
    write_json(out / "training_report.json", result)
    return result
