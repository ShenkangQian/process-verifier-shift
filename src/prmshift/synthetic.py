from pathlib import Path
import random
from .data import question_id
from .io import write_jsonl, write_json


def make_synthetic(output, seed=2026, train_questions=24, test_questions=12, candidates=4):
    """Arithmetic mechanism study with known generative labels; not PRM800K.

    Step labels come from each arithmetic equality. Final-answer labels are
    computed independently. Some trajectories correct an earlier mistake, so
    assigning the final label to every step would be observably incorrect.
    """
    out = Path(output)
    rng = random.Random(seed)
    pairs = [(a, b) for a in range(1, 20) for b in range(1, 20)]
    rng.shuffle(pairs)
    splits = {"train": train_questions, "validation": max(4, test_questions // 2), "test": test_questions}
    offset = 0
    manifest = {"source": "synthetic_arithmetic_mechanism", "seed": seed, "splits": {},
                "limitations": "Short generated arithmetic; no evidence about pretrained mathematical reasoning, PRM800K performance or unseen real LLM generators."}
    for split, count in splits.items():
        questions, steps, cache = [], [], []
        generators = ["synthetic_A", "synthetic_B"] + (["synthetic_C_heldout"] if split == "test" else [])
        for qi, (a, b) in enumerate(pairs[offset:offset + count]):
            problem = f"What is {a}+{b}?"
            qid = question_id(problem)
            total = a + b
            questions.append({"qid": qid, "problem": problem, "answer": str(total), "split": split,
                              "source": "synthetic_arithmetic_mechanism"})
            # Independent local step supervision, not copied from candidate outcomes.
            for delta in (0, 1, -1):
                text = f"{a}+{b}={total + delta}."
                steps.append({"qid": qid, "problem": problem, "prefix": [], "step": text,
                              "label": int(delta == 0), "raw_rating": 1 if delta == 0 else -1,
                              "split": split, "source": "synthetic_arithmetic_mechanism"})
            for gi, generator in enumerate(generators):
                for ci in range(candidates):
                    final_correct = ci == ((qi + gi) % candidates)
                    if qi % 9 == 0: final_correct = False  # Real allwrong pool within this mechanism.
                    if qi % 11 == 1: final_correct = True
                    first_correct = final_correct if ci % 3 else not final_correct
                    val = total if first_correct else total + 1
                    final = total if final_correct else total + 2
                    style = "" if gi == 0 else ("Check: " if gi == 1 else "Compute: ")
                    candidate_steps = [f"{style}{a}+{b}={val}.", f"Answer: {final}"]
                    label = final_correct
                    if split == "test" and qi % 5 == 2 and ci == 0:
                        candidate_steps[-1] = "Answer: unsupported_symbol"
                        label = None
                    cache.append({"qid": qid, "problem": problem, "split": split, "generator": generator,
                                  "candidate_id": f"{ci:04d}", "steps": candidate_steps, "label": label,
                                  "source": "synthetic_arithmetic_mechanism", "synthetic_step_truth": [first_correct, final_correct]})
        offset += count
        write_jsonl(out / f"questions_{split}.jsonl", questions)
        write_jsonl(out / f"steps_{split}.jsonl", steps)
        write_jsonl(out / f"candidates_{split}.jsonl", cache)
        manifest["splits"][split] = {"questions": count, "steps": len(steps), "candidates": len(cache)}
    write_json(out / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Spurious-cue mechanism study (P1/P2/P3). The original make_synthetic above is
# kept unchanged; this generator is a separate dataset.
# ---------------------------------------------------------------------------

CUE_TOKEN = " (verified)"
CUE_RULE = ("The first step text ends with the suffix ' (verified)'. The suffix is chosen per candidate from "
            "the candidate's final-answer correctness only: with sign s=+1 it is present with probability "
            "0.5+kappa when the final answer is correct and 0.5-kappa when it is incorrect; s=-1 swaps the two "
            "probabilities. The suffix never changes the arithmetic content.")
ERROR_DELTAS = (1, -1, 2, -2, 10, -10)

# coverage_target is the per-candidate probability that a candidate's final
# answer is correct, before the forced all-wrong / all-correct pools. Realised
# pool coverage for each N is measured and written to the manifest.
# late_error_rate: among incorrect candidates, the fraction whose first step is
# correct and whose error appears only in the second step. repair_rate: among
# correct candidates, the fraction whose first step is wrong but whose second
# step uses the true intermediate value (wrong step, then repaired).
CUE_GENERATORS = {
    "gen_A": {"coverage_target": 0.6, "kappa": 0.35, "sign": 1, "splits": ("train", "validation", "test"),
              "late_error_rate": 0.5, "repair_rate": 0.25},
    "gen_B": {"coverage_target": 0.25, "kappa": 0.2, "sign": 1, "splits": ("train", "validation", "test"),
              "late_error_rate": 0.1, "repair_rate": 0.1},
    "gen_C_heldout": {"coverage_target": 0.45, "kappa": 0.3, "sign": 1, "splits": ("test",),
                      "late_error_rate": 0.3, "repair_rate": 0.15},
    "gen_D_flipped": {"coverage_target": 0.45, "kappa": 0.35, "sign": -1, "splits": ("test",),
                      "late_error_rate": 0.3, "repair_rate": 0.15},
}


def _wrong_value(rng, total):
    while True:
        value = total + rng.choice(ERROR_DELTAS)
        if value >= 0 and value != total:
            return value


def _first_step(a, b, value, cue):
    return f"{a}+{b}={value}." + (CUE_TOKEN if cue else "")


def _second_step(operand, c, value, answer):
    return f"{operand}+{c}={value}. Answer: {answer}."


def _contingency():
    return {"cue_correct": 0, "cue_incorrect": 0, "nocue_correct": 0, "nocue_incorrect": 0}


def _count(table, cue, correct):
    table[("cue_" if cue else "nocue_") + ("correct" if correct else "incorrect")] += 1


def _rates(table):
    correct = table["cue_correct"] + table["nocue_correct"]
    incorrect = table["cue_incorrect"] + table["nocue_incorrect"]
    p_cue_correct = table["cue_correct"] / correct if correct else None
    p_cue_incorrect = table["cue_incorrect"] / incorrect if incorrect else None
    realised = None
    if p_cue_correct is not None and p_cue_incorrect is not None:
        realised = (p_cue_correct - p_cue_incorrect) / 2
    return {"counts": dict(table), "p_cue_given_correct": p_cue_correct,
            "p_cue_given_incorrect": p_cue_incorrect, "realised_signed_kappa": realised}


def make_synthetic_cue(output, seed=2026, train_questions=300, validation_questions=60, test_questions=200,
                       candidates=16, generators=None, digit_range=(1, 9), allwrong_fraction=0.1,
                       allcorrect_fraction=0.05, pool_sizes=(2, 4, 8, 16), unknown_question_modulus=10):
    """Three-operand single-digit addition pools with a spurious stylistic cue.

    Question: "What is a+b+c?". Candidate steps: "a+b=s1." (optionally with
    the cue suffix) and "s+c=s2. Answer: w.". Each step's label is the local
    arithmetic truth of its own text (s1 == a+b; s+c == s2 and w == s2), and the
    final label is w == a+b+c, computed independently. Candidates include wrong
    first steps that are repaired, propagated early errors whose second step is
    locally true, late errors, all-wrong and all-correct pools. The cue suffix
    is sampled from the final correctness with generator-specific (kappa, sign).
    Step supervision (steps_*.jsonl) has the cue exactly balanced across labels.
    """
    from .objectives import build_pools
    out = Path(output)
    generators = dict(generators or CUE_GENERATORS)
    rng = random.Random(seed)
    low, high = digit_range
    triples = [(a, b, c) for a in range(low, high + 1) for b in range(low, high + 1) for c in range(low, high + 1)]
    rng.shuffle(triples)
    splits = {"train": train_questions, "validation": validation_questions, "test": test_questions}
    if sum(splits.values()) > len(triples):
        raise ValueError("Not enough distinct arithmetic questions for the requested split sizes")
    if candidates < max(pool_sizes):
        raise ValueError("candidates per generator must cover the largest pool size")
    manifest = {"source": "synthetic_arithmetic_cue", "seed": seed, "cue_token": CUE_TOKEN, "cue_rule": CUE_RULE,
                "question_form": "What is a+b+c? with single digits; step 1 'a+b=s1.' (+cue), step 2 's+c=s2. Answer: w.'",
                "step_label_rule": "local arithmetic truth of the step text: step 1 s1==a+b; step 2 s+c==s2 and w==s2",
                "final_label_rule": "w == a+b+c, independent of step labels",
                "candidates_per_question_generator": candidates, "pool_sizes": list(pool_sizes),
                "pool_order": "build_pools shuffles each question x generator once with stable_seed(seed, qid, generator); N pools are prefixes",
                "pool_seed": seed, "digit_range": list(digit_range), "error_deltas": list(ERROR_DELTAS),
                "allwrong_fraction": allwrong_fraction, "allcorrect_fraction": allcorrect_fraction,
                "generators": {g: {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()} for g, spec in generators.items()},
                "splits": {}, "cue_contingency": {}, "realised_coverage": {}, "step_supervision_cue_contingency": {},
                "candidate_structure": {},
                "limitations": "Short generated arithmetic with an artificial stylistic cue; no evidence about pretrained mathematical reasoning, PRM800K or real LLM generators."}
    offset = 0
    for split, count in splits.items():
        questions, steps, cache = [], [], []
        step_table = _contingency()
        tables = {g: _contingency() for g, spec in generators.items() if split in spec["splits"]}
        structure = {g: {"correct_clean": 0, "correct_repaired": 0, "incorrect_late": 0, "incorrect_propagated": 0}
                     for g in tables}
        unknown_total = 0
        for qi, (a, b, c) in enumerate(triples[offset:offset + count]):
            problem = f"What is {a}+{b}+{c}?"
            qid = question_id(problem)
            s1_true = a + b
            total = a + b + c
            questions.append({"qid": qid, "problem": problem, "answer": str(total), "split": split,
                              "source": "synthetic_arithmetic_cue"})
            # Step supervision: exactly balanced cue x label, independent of any candidate.
            for correct in (True, False):
                for cue in (False, True):
                    value = s1_true if correct else _wrong_value(rng, s1_true)
                    steps.append({"qid": qid, "problem": problem, "prefix": [], "step": _first_step(a, b, value, cue),
                                  "label": int(correct), "raw_rating": 1 if correct else -1, "cue": cue,
                                  "step_type": "first", "split": split, "source": "synthetic_arithmetic_cue"})
                    _count(step_table, cue, correct)
            prefix_cue = qi % 2 == 0
            prefix_value = s1_true if rng.random() < 0.5 else _wrong_value(rng, s1_true)
            operand = s1_true if rng.random() < 0.7 else _wrong_value(rng, s1_true)
            for correct in (True, False):
                if correct:
                    value = operand + c
                    answer = value
                elif rng.random() < 0.5:
                    value = _wrong_value(rng, operand + c)
                    answer = value
                else:
                    value = operand + c
                    answer = _wrong_value(rng, value)
                steps.append({"qid": qid, "problem": problem, "prefix": [_first_step(a, b, prefix_value, prefix_cue)],
                              "step": _second_step(operand, c, value, answer), "label": int(correct),
                              "raw_rating": 1 if correct else -1, "cue": prefix_cue, "step_type": "second",
                              "split": split, "source": "synthetic_arithmetic_cue"})
                _count(step_table, prefix_cue, correct)
            for generator, spec in generators.items():
                if split not in spec["splits"]:
                    continue
                mode = rng.random()
                if mode < allwrong_fraction:
                    outcomes = [False] * candidates
                elif mode < allwrong_fraction + allcorrect_fraction:
                    outcomes = [True] * candidates
                else:
                    outcomes = [rng.random() < spec["coverage_target"] for _ in range(candidates)]
                for ci, final_correct in enumerate(outcomes):
                    if final_correct:
                        if rng.random() < spec["repair_rate"]:
                            s1, kind = _wrong_value(rng, s1_true), "correct_repaired"
                        else:
                            s1, kind = s1_true, "correct_clean"
                        operand, s2, answer = s1_true, total, total
                    elif rng.random() < spec["late_error_rate"]:
                        s1, kind = s1_true, "incorrect_late"
                        operand = s1_true
                        if rng.random() < 0.5:
                            s2 = _wrong_value(rng, total)
                            answer = s2
                        else:
                            s2 = total
                            answer = _wrong_value(rng, total)
                    else:
                        s1, kind = _wrong_value(rng, s1_true), "incorrect_propagated"
                        operand = s1
                        s2 = s1 + c
                        answer = s2
                    structure[generator][kind] += 1
                    step_truth = [s1 == s1_true, operand + c == s2 and answer == s2]
                    p_cue = 0.5 + spec["sign"] * spec["kappa"] if final_correct else 0.5 - spec["sign"] * spec["kappa"]
                    cue = rng.random() < p_cue
                    candidate_steps = [_first_step(a, b, s1, cue), _second_step(operand, c, s2, answer)]
                    label = final_correct
                    assert label == (answer == total)
                    if split == "test" and qi % unknown_question_modulus == 3 and ci == 5:
                        candidate_steps[-1] = f"{operand}+{c}={s2}. Answer: unsupported_symbol."
                        label = None
                        step_truth[1] = None  # The answer claim is not checkable from the text.
                        unknown_total += 1
                    else:
                        _count(tables[generator], cue, final_correct)
                    cache.append({"qid": qid, "problem": problem, "split": split, "generator": generator,
                                  "candidate_id": f"{ci:04d}", "steps": candidate_steps, "label": label, "cue": cue,
                                  "source": "synthetic_arithmetic_cue", "candidate_kind": kind,
                                  "synthetic_step_truth": step_truth})
        offset += count
        write_jsonl(out / f"questions_{split}.jsonl", questions)
        write_jsonl(out / f"steps_{split}.jsonl", steps)
        write_jsonl(out / f"candidates_{split}.jsonl", cache)
        manifest["splits"][split] = {"questions": count, "steps": len(steps), "candidates": len(cache),
                                     "generators": sorted(tables), "unknown_labels": unknown_total}
        manifest["cue_contingency"][split] = {g: _rates(t) for g, t in tables.items()}
        manifest["step_supervision_cue_contingency"][split] = _rates(step_table)
        manifest["candidate_structure"][split] = structure
        pools, _ = build_pools(cache, list(pool_sizes), seed, training=False)
        coverage = {}
        for pool in pools:
            entry = coverage.setdefault(pool["group"], {"pools": 0, "known_correct": 0, "possibly_correct": 0,
                                                        "all_correct": 0, "all_incorrect": 0})
            labels = [c["label"] for c in pool["candidates"]]
            entry["pools"] += 1
            entry["known_correct"] += int(any(x is True for x in labels))
            entry["possibly_correct"] += int(any(x is not False for x in labels))
            entry["all_correct"] += int(all(x is True for x in labels))
            entry["all_incorrect"] += int(all(x is False for x in labels))
        for group, entry in coverage.items():
            entry["coverage_lower"] = entry["known_correct"] / entry["pools"]
            entry["coverage_upper"] = entry["possibly_correct"] / entry["pools"]
        manifest["realised_coverage"][split] = dict(sorted(coverage.items()))
    write_json(out / "manifest.json", manifest)
    return manifest
