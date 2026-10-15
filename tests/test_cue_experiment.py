import re
from collections import Counter, defaultdict
import pytest
import torch
from prmshift.synthetic import make_synthetic_cue, CUE_GENERATORS, CUE_TOKEN
from prmshift.objectives import build_pools, soft_selection_loss, hard_selection_error, pool_selection_metrics
from prmshift.evaluate import aggregate, pool_outcome, worst_groups, evaluate
from prmshift.io import read_jsonl
from prmshift.train import train

torch.set_num_threads(2)


@pytest.fixture(scope="module")
def cue_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("cue")
    manifest = make_synthetic_cue(out, seed=11, train_questions=60, validation_questions=12, test_questions=60)
    return out, manifest


def _parse_first(step):
    m = re.fullmatch(r"(\d+)\+(\d+)=(\d+)\.(?: \(verified\))?", step)
    return tuple(int(x) for x in m.groups()), step.endswith(CUE_TOKEN)


def _parse_second(step):
    m = re.fullmatch(r"(\d+)\+(\d+)=(\d+)\. Answer: (\S+)\.", step)
    s, c, v, w = m.groups()
    return int(s), int(c), int(v), (int(w) if w.isdigit() else None)


def test_cue_contingency_matches_kappa_and_sign(cue_data):
    _, manifest = cue_data
    for split, tables in manifest["cue_contingency"].items():
        for generator, rates in tables.items():
            spec = CUE_GENERATORS[generator]
            assert split in spec["splits"]
            target = spec["sign"] * spec["kappa"]
            assert rates["realised_signed_kappa"] == pytest.approx(target, abs=0.07), (split, generator, rates)
            expected_p_correct = 0.5 + target
            assert rates["p_cue_given_correct"] == pytest.approx(expected_p_correct, abs=0.1)
            assert rates["p_cue_given_incorrect"] == pytest.approx(1 - expected_p_correct, abs=0.1)
    assert manifest["cue_contingency"]["test"]["gen_D_flipped"]["realised_signed_kappa"] < 0
    assert manifest["cue_contingency"]["train"]["gen_A"]["realised_signed_kappa"] > 0
    assert set(manifest["cue_contingency"]["train"]) == {"gen_A", "gen_B"}
    assert set(manifest["cue_contingency"]["test"]) == {"gen_A", "gen_B", "gen_C_heldout", "gen_D_flipped"}


def test_every_question_has_every_generator_and_pools_are_prefixes(cue_data):
    out, manifest = cue_data
    for split, expected in (("train", {"gen_A", "gen_B"}), ("test", {"gen_A", "gen_B", "gen_C_heldout", "gen_D_flipped"})):
        rows = list(read_jsonl(out / f"candidates_{split}.jsonl"))
        per_question = defaultdict(Counter)
        for r in rows:
            per_question[r["qid"]][r["generator"]] += 1
        assert len(per_question) == manifest["splits"][split]["questions"]
        for counts in per_question.values():
            assert set(counts) == expected
            assert all(v == 16 for v in counts.values())
        pools, _ = build_pools(rows, [2, 4, 8, 16], seed=11, training=False)
        by_key = defaultdict(dict)
        for p in pools:
            by_key[(p["qid"], p["generator"])][p["n"]] = [c["candidate_id"] for c in p["candidates"]]
        for ids in by_key.values():
            assert set(ids) == {2, 4, 8, 16}
            for small, large in ((2, 4), (4, 8), (8, 16)):
                assert ids[large][:small] == ids[small]
            assert len(set(ids[16])) == 16
        unknown = sum(r["label"] is None for r in rows)
        assert (unknown > 0) == (split == "test")
        assert unknown == manifest["splits"][split]["unknown_labels"]
        if split == "test":
            assert unknown < 0.05 * len(rows)


def test_step_labels_are_local_arithmetic_and_independent_of_cue(cue_data):
    out, manifest = cue_data
    for split in ("train", "validation", "test"):
        rates = manifest["step_supervision_cue_contingency"][split]
        assert rates["p_cue_given_correct"] == 0.5 and rates["p_cue_given_incorrect"] == 0.5
        assert rates["realised_signed_kappa"] == 0.0
        table = Counter()
        for row in read_jsonl(out / f"steps_{split}.jsonl"):
            if row["prefix"]:
                s, c, v, w = _parse_second(row["step"])
                assert row["label"] == int(s + c == v and w == v)
                assert row["cue"] == row["prefix"][0].endswith(CUE_TOKEN)
            else:
                (a, b, v), cue = _parse_first(row["step"])
                assert f"What is {a}+{b}+" in row["problem"]
                assert row["label"] == int(a + b == v)
                assert cue == row["cue"]
            table[(row["cue"], row["label"])] += 1
        assert table[(True, 1)] == table[(False, 1)] == table[(True, 0)] == table[(False, 0)]
    # Candidate labels: step truth from the text, final label from a+b+c only.
    kinds = Counter()
    for r in read_jsonl(out / "candidates_test.jsonl"):
        a, b, c = map(int, re.findall(r"\d+", r["problem"]))
        (pa, pb, s1), cue = _parse_first(r["steps"][0])
        s, cc, v, w = _parse_second(r["steps"][1])
        assert (pa, pb, cc) == (a, b, c) and cue == r["cue"]
        assert r["synthetic_step_truth"] == [s1 == a + b, None if w is None else (s + c == v and w == v)]
        assert r["label"] == (None if w is None else w == a + b + c)
        kinds[r["candidate_kind"]] += 1
    assert set(kinds) == {"correct_clean", "correct_repaired", "incorrect_late", "incorrect_propagated"}
    assert "gen_A::n=2" in manifest["realised_coverage"]["test"]
    assert manifest["realised_coverage"]["test"]["gen_B::n=2"]["coverage_lower"] < manifest["realised_coverage"]["test"]["gen_A::n=2"]["coverage_lower"]


def _outcome(labels, scores, qid, group):
    cs = [{"qid": qid, "generator": group.split("::")[0], "candidate_id": str(i), "label": y, "score": s}
          for i, (y, s) in enumerate(zip(labels, scores))]
    return pool_outcome({"qid": qid, "group": group, "generator": group.split("::")[0], "n": len(cs), "candidates": cs})


def test_unconditional_risk_identity_and_margins():
    rows = [_outcome([True, False], [1, 2], "a", "g::n=2"), _outcome([False, False], [1, 2], "b", "g::n=2"),
            _outcome([True, False], [2, 1], "c", "g::n=2"), _outcome([False, True], [3, 1], "d", "g::n=2")]
    report = aggregate(rows)
    assert report["all_labels_known"]
    assert report["coverage"] == pytest.approx(0.75)
    assert report["conditional_failure"] == pytest.approx(2 / 3)
    assert report["unconditional_risk"] == pytest.approx(report["coverage"] * report["conditional_failure"])
    assert report["overall_accuracy"] == pytest.approx(report["coverage"] * (1 - report["conditional_failure"]))
    assert report["overall_accuracy"] == pytest.approx(report["coverage"] - report["unconditional_risk"])
    assert report["error_rate"] == pytest.approx(1 - report["overall_accuracy"])
    assert report["unconditional_risk_bounds"] == [report["unconditional_risk"]] * 2
    assert rows[0]["margin"] == 1 and rows[0]["top1_correct"] is False
    assert rows[2]["margin"] == 1 and rows[2]["top1_correct"] is True
    partial = aggregate([_outcome([True, None], [1, 2], "e", "g::n=2")])
    assert partial["unconditional_risk"] is None and partial["unconditional_risk_bounds"] == [0.0, 1.0]


def test_worst_group_reversal_between_criteria():
    # Low coverage, low conditional failure vs high coverage, high conditional failure.
    low = [_outcome([True, False], [2, 1], f"l{i}", "low::n=2") for i in range(2)] + \
          [_outcome([False, False], [1, 2], f"l{i}", "low::n=2") for i in range(2, 10)]
    high = [_outcome([True, False], [1, 2], f"h{i}", "high::n=2") for i in range(5)] + \
           [_outcome([True, False], [2, 1], f"h{i}", "high::n=2") for i in range(5, 10)]
    reports = {"low::n=2": aggregate(low), "high::n=2": aggregate(high)}
    assert reports["low::n=2"]["conditional_failure"] == 0.0 and reports["high::n=2"]["conditional_failure"] == 0.5
    table = worst_groups(reports)
    assert table["conditional"]["group"] == "high::n=2"
    assert table["error_rate"]["group"] == "low::n=2"
    assert table["unconditional"]["group"] == "high::n=2"
    assert table["conditional"]["robust"]
    assert table["reversal"] is False
    swapped = {"low::n=2": aggregate(low), "high::n=2": aggregate([_outcome([True, False], [2, 1], f"h{i}", "high::n=2") for i in range(10)])}
    table = worst_groups(swapped)
    assert table["conditional"]["group"] == "low::n=2" or table["conditional"]["value_bounds"] == [0.0, 0.0]


def test_hard_error_bounded_by_n_times_soft_loss():
    torch.manual_seed(3)
    for _ in range(50):
        n = int(torch.randint(2, 9, (1,)))
        scores = torch.randn(n)
        labels = [bool(x) for x in torch.rand(n) < 0.4]
        if not any(labels):
            labels[0] = True
        metrics = pool_selection_metrics(scores, labels, 0.5)
        assert metrics["hard"] in (0.0, 1.0)
        assert metrics["hard"] <= n * metrics["soft"] + 1e-6
        selected = int(scores.argmax())
        assert metrics["hard"] == float(not labels[selected])
    with pytest.raises(ValueError):
        hard_selection_error(torch.zeros(2), [True, None])


@pytest.mark.parametrize("method", ["ce", "group_dro"])
def test_training_logs_soft_and_hard_per_group_and_validation(tmp_path, method):
    make_synthetic_cue(tmp_path / "data", seed=5, train_questions=12, validation_questions=4, test_questions=4)
    config = {"seed": 5, "method": method, "steps": str(tmp_path / "data/steps_train.jsonl"),
              "candidates": str(tmp_path / "data/candidates_train.jsonl"),
              "validation_steps": str(tmp_path / "data/steps_validation.jsonl"),
              "validation_candidates": str(tmp_path / "data/candidates_validation.jsonl"),
              "output": str(tmp_path / method), "iterations": 3, "validate_every": 1, "step_batch_size": 4,
              "pool_sizes": [2, 4], "learning_rate": .01, "temperature": 0.5,
              "model": {"kind": "tiny", "width": 16, "heads": 2, "max_length": 256}}
    report = train(config)
    groups = {"gen_A::n=2", "gen_A::n=4", "gen_B::n=2", "gen_B::n=4"}
    for entry in report["history"]:
        assert set(entry["selection_soft_loss"]) == groups and set(entry["selection_hard_error"]) == groups
        for g in groups:
            n = int(g.split("n=")[1])
            assert entry["selection_hard_error"][g] in (0.0, 1.0)
            assert entry["selection_hard_error"][g] <= n * entry["selection_soft_loss"][g] + 1e-6
        if method == "ce":
            assert entry["selection_risk"] == {}
        else:
            assert entry["selection_risk"] == pytest.approx(entry["selection_soft_loss"])
    assert [v["iteration"] for v in report["validation_curve"]] == [0, 1, 2, 3]
    for v in report["validation_curve"]:
        assert set(v["groups"]) == groups
        for g, m in v["groups"].items():
            assert 0 <= m["hard"] <= 1 and 0 <= m["soft"] <= 1 and m["pools"] > 0
        assert 0 <= v["step_accuracy"] <= 1 and v["labelled_steps"] == 24
    assert report["cost"]["diagnostic_forward_tokens"] > 0
    result = evaluate(config["output"], tmp_path / "data/candidates_test.jsonl", tmp_path / "evaluation", [2, 4], bootstrap=5)
    assert set(result["generators"]) == {"gen_A", "gen_B", "gen_C_heldout", "gen_D_flipped"}
    assert set(result["worst_groups"]) == {"by_group", "by_generator"}
    for row in read_jsonl(tmp_path / "evaluation/pool_outcomes.jsonl"):
        assert row["top1_correct"] in (True, False, None)
        assert row["margin"] is not None and row["margin"] >= 0
        assert row["margin"] == pytest.approx(row["top1_score"] - row["top2_score"])
