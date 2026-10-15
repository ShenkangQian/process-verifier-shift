from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import random
import math
import requests
from .io import read_jsonl, write_jsonl, write_json, sha256

PRM_REVISION = "7ecc794703b2877f63226f2477a49b34f9b25163"
PRM_FILES = {f"phase{phase}_{split}.jsonl": f"prm800k/data/phase{phase}_{split}.jsonl" for phase in (1, 2) for split in ("train", "test")}
PRM_FILES.update({f"math_{split}.jsonl": f"prm800k/math_splits/{split}.jsonl" for split in ("train", "test")})


def canonical_problem(problem):
    # Preserve numbers: replacing them would conflate distinct mathematics.
    text = str(problem).casefold().replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\,", "").replace(r"\!", "").replace("$", "")
    return re.sub(r"\s+", "", text)


def question_id(problem):
    return hashlib.sha256(canonical_problem(problem).encode()).hexdigest()[:24]


def download_prm(directory, files, revision=PRM_REVISION, max_records=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in files:
        if name not in PRM_FILES:
            raise ValueError(f"unknown official file {name}")
        url = f"https://media.githubusercontent.com/media/openai/prm800k/{revision}/{PRM_FILES[name]}"
        target = directory / (name if max_records is None else f"sample_{name}")
        tmp = target.with_suffix(target.suffix + ".partial")
        count = 0
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with tmp.open("wb") as out:
                for line in response.iter_lines():
                    if not line:
                        continue
                    json.loads(line)  # Reject Git LFS pointer/error pages before saving.
                    out.write(line + b"\n")
                    count += 1
                    if max_records is not None and count >= max_records:
                        break
        tmp.replace(target)
        manifest.append({"file": str(target), "url": url, "revision": revision, "records": count,
                         "partial_sample": max_records is not None, "sha256": sha256(target)})
    write_json(directory / "download_manifest.json", manifest)
    return manifest


def convert_prm_record(record, neutral="mask"):
    """Expand alternatives at the *same* prefix; follow only the chosen trajectory.

    A null chosen completion with no human continuation terminates the known
    prefix. Unannotated pre_generated_steps are never assigned labels.
    """
    if neutral not in ("mask", "positive"):
        raise ValueError("neutral must be mask or positive")
    problem = record["question"]["problem"]
    qid = question_id(problem)
    prefix = []
    for step_index, step in enumerate(record["label"]["steps"]):
        for alternative_index, completion in enumerate(step.get("completions") or []):
            raw = completion.get("rating")
            flagged = completion.get("flagged")
            target = None
            if not flagged:
                if raw == 1: target = 1
                elif raw == -1: target = 0
                elif raw == 0 and neutral == "positive": target = 1
            yield {"qid": qid, "problem": problem, "prefix": list(prefix), "step": completion["text"],
                   "label": target, "raw_rating": raw, "flagged": flagged,
                   "step_index": step_index, "alternative_index": alternative_index,
                   "source": "official_prm800k", "finish_reason": record["label"].get("finish_reason")}
        chosen = step.get("chosen_completion")
        human = step.get("human_completion")
        if chosen is not None:
            completions = step.get("completions") or []
            if not isinstance(chosen, int) or not 0 <= chosen < len(completions):
                raise ValueError("invalid chosen_completion index")
            prefix.append(completions[chosen]["text"])
        elif human is not None:
            prefix.append(human if isinstance(human, str) else human["text"])
        else:
            break


def blocked_problem(record):
    if "question" in record:
        q = record["question"]
        return q["problem"] if isinstance(q, dict) else q
    return record.get("problem")


def prepare_prm(inputs, output, blocked=(), neutral="mask", val_fraction=0.1, seed=2026):
    """Only official *_train inputs; entire question goes to one split.

    Protected evaluation questions take precedence. Near duplicates are reported
    by audit_near_duplicates separately and must be reviewed before GPU runs.
    """
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    banned = {question_id(blocked_problem(r)) for p in blocked for r in read_jsonl(p) if blocked_problem(r)}
    handles = {s: (out / f"steps_{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "validation")}
    stats = Counter()
    questions = {}
    seen_steps = set()
    try:
        for path in inputs:
            if "test" in Path(path).name.lower():
                raise ValueError("official test labels cannot be prepared as training data")
            for record in read_jsonl(path):
                problem = record["question"]["problem"]
                qid = question_id(problem)
                if qid in banned:
                    stats["blocked_records"] += 1
                    continue
                u = int(hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()[:8], 16) / 2**32
                split = "validation" if u < val_fraction else "train"
                questions[qid] = {"qid": qid, "problem": problem, "answer": record["question"].get("ground_truth_answer"), "split": split, "source": "official_prm800k"}
                for step in convert_prm_record(record, neutral):
                    key = (qid, tuple(step["prefix"]), step["step"], step["label"])
                    if key in seen_steps:
                        stats["duplicate_steps"] += 1
                        continue
                    seen_steps.add(key)
                    step["split"] = split
                    handles[split].write(json.dumps(step, ensure_ascii=False) + "\n")
                    stats[f"{split}_steps"] += 1
                    stats["unlabelled_steps" if step["label"] is None else "labelled_steps"] += 1
    finally:
        for handle in handles.values(): handle.close()
    for split in handles:
        write_jsonl(out / f"questions_{split}.jsonl", (q for q in questions.values() if q["split"] == split))
    write_json(out / "prepare_manifest.json", {"statistics": dict(stats), "neutral": neutral, "seed": seed,
        "inputs": [{"path": str(p), "sha256": sha256(p)} for p in inputs], "protected_questions": len(banned),
        "near_duplicate_review_required": True})
    return dict(stats)


def audit_splits(paths, near_threshold=0.9):
    """Deterministic exact + token-shingle near-duplicate review (no LLM judge)."""
    if not 0 < near_threshold <= 1: raise ValueError("near_threshold must be in (0, 1]")
    by_id, rows, exact = {}, [], []
    seen_question_split = set()
    for path in paths:
        for row in read_jsonl(path):
            text = blocked_problem(row)
            if not text: continue
            qid = question_id(text)
            split = row.get("split", Path(path).stem)
            if qid in by_id and split != by_id[qid]:
                exact.append({"qid": qid, "splits": [by_id[qid], split]})
            by_id[qid] = split
            if (qid, split) in seen_question_split: continue
            seen_question_split.add((qid, split))
            rows.append((qid, split, set(re.findall(r"\w+|[^\w\s]", text.casefold()))))
    # Inverted index avoids a full pairwise scan. The review metric is token-set
    # Jaccard, deliberately reported as a candidate flag rather than proof.
    postings, near, seen = {}, [], set()
    frequencies = Counter(token for _, _, tokens in rows for token in tokens)
    for i, (qid, split, tokens) in enumerate(rows):
        candidates = set()
        # If Jaccard >= t, at most (1-t)|tokens| query tokens can be
        # absent. Querying that many + 1 rare tokens preserves candidate recall.
        prefix_count = len(tokens) - math.ceil(near_threshold * len(tokens) - 1e-12) + 1
        query_tokens = sorted(tokens, key=lambda token: (frequencies[token], token))[:prefix_count]
        for token in query_tokens:
            candidates.update(postings.get(token, []))
        for j in candidates:
            oq, os, ot = rows[j]
            key = tuple(sorted((qid, oq)))
            if os == split or qid == oq or key in seen: continue
            if min(len(tokens), len(ot)) < near_threshold * max(len(tokens), len(ot)): continue
            similarity = len(tokens & ot) / max(1, len(tokens | ot))
            if similarity >= near_threshold:
                near.append({"qids": [oq, qid], "splits": [os, split], "similarity": similarity})
                seen.add(key)
        for token in tokens: postings.setdefault(token, []).append(i)
    return {"exact_cross_split": exact, "near_duplicate_review": near, "safe_exact": not exact,
            "note": "Near-duplicate flags require manual review; no claim about unknown backbone pretraining data."}


def budget_prm(input_directory, output_directory, step_budget, seed=2026):
    """Uniform reservoir of explicit training labels, before any model training.

    Budget counts retained explicit step labels, not original human annotation
    effort. Raw text prefixes and public final-answer verification are separate
    resources and must be reported separately.
    """
    source, output = Path(input_directory), Path(output_directory)
    if source.resolve() == output.resolve(): raise ValueError("Budgeted output must be a new directory")
    if step_budget < 1: raise ValueError("step_budget must be positive")
    rng = random.Random(seed)
    reservoir, count = [], 0
    for row in read_jsonl(source / "steps_train.jsonl"):
        if row.get("label") not in (0, 1): continue
        count += 1
        if len(reservoir) < step_budget: reservoir.append(row)
        else:
            index = rng.randrange(count)
            if index < step_budget: reservoir[index] = row
    if count < step_budget:
        raise ValueError(f"Requested {step_budget} labelled steps, only {count} available")
    reservoir.sort(key=lambda r: (r["qid"], r.get("step_index", 0), r.get("alternative_index", 0), r["step"]))
    qids = {r["qid"] for r in reservoir}
    write_jsonl(output / "steps_train.jsonl", reservoir)
    write_jsonl(output / "questions_train.jsonl", (r for r in read_jsonl(source / "questions_train.jsonl") if r["qid"] in qids))
    for name in ("steps_validation.jsonl", "questions_validation.jsonl"):
        write_jsonl(output / name, read_jsonl(source / name))
    result = {"seed": seed, "explicit_step_label_budget": step_budget, "available_labelled_steps": count,
              "retained_training_questions": len(qids), "source_steps_sha256": sha256(source / "steps_train.jsonl"),
              "budget_unit": "explicit step correctness labels retained for training; not annotation hours, prefix provenance, or final-answer verification",
              "same_file_required_for_all_methods": True}
    write_json(output / "budget_manifest.json", result)
    return result
