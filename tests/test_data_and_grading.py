from pathlib import Path
import copy
import json
import pytest
from prmshift.data import convert_prm_record, prepare_prm, question_id, audit_splits, budget_prm
from prmshift.io import read_jsonl, write_jsonl, sha256
from prmshift.grading import extract_answer, grade_answer

FIXTURES = Path(__file__).parent / "fixtures"


def test_actual_official_schema_and_provenance():
    rows = list(read_jsonl(FIXTURES / "sample_phase1_train.jsonl"))
    manifest = json.loads((FIXTURES / "download_manifest.json").read_text())
    assert sha256(FIXTURES / "sample_phase1_train.jsonl") == manifest[0]["sha256"]
    assert manifest[0]["partial_sample"] is True
    assert len(rows) == 2
    assert [len(list(convert_prm_record(r))) for r in rows] == [11, 35]
    second = list(convert_prm_record(rows[1]))
    assert second[0]["label"] is None  # Official neutral, not fabricated.
    assert second[2]["label"] == 1
    assert second[7]["prefix"] == [rows[1]["label"]["steps"][0]["completions"][6]["text"]]
    assert all(r["source"] == "official_prm800k" for r in second)


def test_actual_phase2_schema_does_not_label_unseen_tail():
    rows = list(read_jsonl(FIXTURES / "sample_phase2_train.jsonl"))
    assert len(rows) == 2
    for record in rows:
        converted = list(convert_prm_record(record))
        assert converted
        assert max(r["step_index"] for r in converted) < len(record["label"]["steps"])
        allowed = {c["text"] for s in record["label"]["steps"] for c in s["completions"]}
        assert all(r["step"] in allowed for r in converted)


def test_branch_masking_and_null_trajectory_stops():
    record = {"question": {"problem": "q"}, "label": {"steps": [
        {"completions": [{"text": "A", "rating": 1}, {"text": "B", "rating": -1}], "chosen_completion": 1},
        {"completions": [{"text": "C", "rating": 1, "flagged": True}], "chosen_completion": None},
        {"completions": [{"text": "future", "rating": 1}], "chosen_completion": 0}]}}
    rows = list(convert_prm_record(record))
    assert len(rows) == 3
    assert rows[0]["prefix"] == rows[1]["prefix"] == []
    assert rows[2]["prefix"] == ["B"]
    assert rows[2]["label"] is None


def test_protected_question_exclusion_and_question_split(tmp_path):
    fixture = FIXTURES / "sample_phase1_train.jsonl"
    rows = list(read_jsonl(fixture))
    blocked = tmp_path / "protected.jsonl"
    write_jsonl(blocked, [{"problem": rows[0]["question"]["problem"]}])
    stats = prepare_prm([fixture], tmp_path / "prepared", [blocked], val_fraction=.5)
    assert stats["blocked_records"] == 1
    train = list(read_jsonl(tmp_path / "prepared/questions_train.jsonl"))
    val = list(read_jsonl(tmp_path / "prepared/questions_validation.jsonl"))
    assert not ({r["qid"] for r in train} & {r["qid"] for r in val})
    assert len(train) + len(val) == 1


def test_exact_leakage_audit(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, [{"problem": "What is $2+2$?", "split": "train"}])
    write_jsonl(b, [{"problem": "What is 2+2?", "split": "test"}])
    assert not audit_splits([a, b])["safe_exact"]
    # At exactly 0.9 Jaccard, one rare token may be absent. Floating-point
    # prefix rounding must not suppress the review candidate.
    write_jsonl(a, [{"problem": "a b c d e f g h i j", "split": "train"}])
    write_jsonl(b, [{"problem": "b c d e f g h i j", "split": "test"}])
    assert len(audit_splits([a, b])["near_duplicate_review"]) == 1


def test_fixed_label_budget_reproducible(tmp_path):
    from prmshift.synthetic import make_synthetic
    make_synthetic(tmp_path / "input", train_questions=10, test_questions=2)
    a = budget_prm(tmp_path / "input", tmp_path / "a", 7, 17)
    budget_prm(tmp_path / "input", tmp_path / "b", 7, 17)
    assert a["explicit_step_label_budget"] == 7
    assert len(list(read_jsonl(tmp_path / "a/steps_train.jsonl"))) == 7
    assert sha256(tmp_path / "a/steps_train.jsonl") == sha256(tmp_path / "b/steps_train.jsonl")
    with pytest.raises(ValueError): budget_prm(tmp_path / "input", tmp_path / "c", 1000)


@pytest.mark.parametrize("pred,ref,label", [(r"\frac{1}{2}", "0.5", True), ("2", "3", False),
    (None, "3", None), ("x+x", "2x", None), ("__import__('os').system('echo secret')", "2", None),
    ("1/0", "2", None), ("1,000", "1000", True),
    ("0.100000000000000000001", "0.1", False), ("1e99999999999999999999999", "1", None)])
def test_conservative_grading(pred, ref, label):
    assert grade_answer(pred, ref)["label"] is label


def test_balanced_box_extraction():
    assert extract_answer(r"Old \boxed{3}. New \boxed{\frac{1}{2}}.") == r"\frac{1}{2}"
    assert extract_answer("There is no explicit final answer.") is None
