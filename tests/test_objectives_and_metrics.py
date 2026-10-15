import torch
import pytest
from prmshift.objectives import build_pools, soft_selection_loss, GroupDRO
from prmshift.evaluate import pool_outcome, aggregate, summarize


def candidates(labels, scores=None, qid="q"):
    scores = scores or list(range(len(labels)))
    return [{"qid": qid, "generator": "g", "candidate_id": str(i), "label": y, "score": s}
            for i, (y, s) in enumerate(zip(labels, scores))]


def outcome(labels, scores=None, qid="q"):
    cs = candidates(labels, scores, qid)
    return pool_outcome({"qid": qid, "group": f"g::n={len(cs)}", "n": len(cs), "candidates": cs})


def test_condition_and_unknown_are_not_negative():
    cs = candidates([True, False, None, False])
    pools, stats = build_pools(cs, [2, 4], training=True)
    assert all(p["n"] == 2 for p in pools)
    assert stats["unknown_candidates_removed_for_training"] == 1
    assert all(c["label"] is not None for p in pools for c in p["candidates"])
    assert build_pools(candidates([False, False]), [2], training=True)[0] == []
    with pytest.raises(ValueError): soft_selection_loss(torch.zeros(2), [False, False])
    with pytest.raises(ValueError): soft_selection_loss(torch.zeros(2), [True, None])


def test_soft_loss_gradient_and_hard_bound():
    scores = torch.tensor([1., 2., 3.], requires_grad=True)
    loss = soft_selection_loss(scores, [True, False, False])
    loss.backward()
    assert scores.grad[0] < 0 and scores.grad[2] > 0
    assert 1 <= 3 * loss
    assert soft_selection_loss(scores, [True, True, True]) == 0


def test_dro_increases_highrisk_group_weight():
    dro = GroupDRO(["a", "b"], eta=1)
    dro.update({"a": torch.tensor(.1), "b": torch.tensor(.9)})
    assert dro.weights()[1] > dro.weights()[0]
    other = GroupDRO(["a", "b"])
    other.load_state_dict(dro.state_dict())
    assert torch.equal(other.weights(), dro.weights())


def test_coverage_identity_complete_labels():
    rows = [outcome([True, False], [1, 2], "a"), outcome([False, False], [1, 2], "b"),
            outcome([True, False], [2, 1], "c")]
    report = aggregate(rows)
    assert report["coverage"] == pytest.approx(2/3)
    assert report["conditional_failure"] == .5
    assert report["overall_accuracy"] == pytest.approx(report["coverage"] * (1-report["conditional_failure"]))


def test_unknown_bounds_do_not_invent_labels():
    row = outcome([True, None], [1, 2])
    report = aggregate([row])
    assert report["overall_accuracy"] is None
    assert report["conditional_failure_bounds"] == [0, 1]
    assert aggregate([outcome([None], [1])])["conditional_failure_bounds"] == [0, 0]


def test_question_bootstrap_reproducible():
    rows = [outcome([True, False], [1, 2], "a"), outcome([True, False], [2, 1], "b")]
    assert summarize(rows, 20, 1) == summarize(rows, 20, 1)
    assert summarize(rows, 20)["unique_questions"] == 2
