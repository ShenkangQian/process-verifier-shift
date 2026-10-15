from __future__ import annotations
import random
from collections import defaultdict, Counter
import torch
from .io import stable_seed


def build_pools(candidates, sizes, seed=2026, training=False):
    """Fixed nested pools, one question per generator×N; no replacement.

    Train removes unknown labels BEFORE selecting a pool, so group N is the
    actual retained count. Evaluation preserves unknown candidates. An all-wrong
    set is excluded only from the conditional training objective.
    """
    by_question = defaultdict(list)
    seen = set()
    stats = Counter()
    for c in candidates:
        if c.get("label") is not None and type(c["label"]) is not bool:
            raise ValueError("Candidate label must be bool or null, not a step rating")
        key = (c["qid"], c["generator"], c["candidate_id"])
        if key in seen: raise ValueError(f"Duplicate candidate identity {key}")
        seen.add(key)
        if training and c.get("label") is None:
            stats["unknown_candidates_removed_for_training"] += 1
            continue
        by_question[(c["qid"], c["generator"])].append(c)
    pools = []
    for (qid, generator), rows in sorted(by_question.items()):
        rows = sorted(rows, key=lambda r: str(r["candidate_id"]))
        random.Random(stable_seed(seed, qid, generator)).shuffle(rows)
        for n in sizes:
            if len(rows) < n:
                stats["insufficient_candidates"] += 1
                continue
            selected = rows[:n]
            if training and not any(c.get("label") is True for c in selected):
                stats["allwrong_pools_excluded_from_conditional_training"] += 1
                continue
            pools.append({"qid": qid, "generator": generator, "n": n,
                          "group": f"{generator}::n={n}", "candidates": selected})
    stats["pools"] = len(pools)
    return pools, dict(stats)


def soft_selection_loss(scores, labels, temperature=1.0):
    if temperature <= 0: raise ValueError("temperature must be positive")
    if any(type(x) is not bool for x in labels):
        raise ValueError("unknown labels cannot be converted to incorrect")
    if not any(labels):
        raise ValueError("conditional objective requires a correct candidate")
    wrong = torch.tensor([not x for x in labels], device=scores.device, dtype=scores.dtype)
    return (scores.div(temperature).softmax(0) * wrong).sum()


def hard_selection_error(scores, labels):
    """1.0 if the argmax candidate is incorrect, else 0.0 (first index on ties).

    Proposition P1: e <= N * soft_selection_loss for the same pool, because the
    argmax has softmax mass >= 1/N and contributes to the soft loss when wrong.
    """
    if any(type(x) is not bool for x in labels):
        raise ValueError("hard selection error needs fully known labels")
    selected = int(torch.as_tensor(scores).argmax())
    return 0.0 if labels[selected] else 1.0


def pool_selection_metrics(scores, labels, temperature=1.0):
    """Soft loss and hard error for one eligible pool; requires a correct candidate."""
    scores = torch.as_tensor(scores, dtype=torch.float32)
    return {"soft": float(soft_selection_loss(scores, labels, temperature)),
            "hard": hard_selection_error(scores, labels), "n": len(labels)}


class GroupDRO:
    """Exponentiated-gradient adversary on observed training groups only."""
    def __init__(self, groups, eta=0.1):
        self.groups = sorted(groups)
        self.eta = eta
        self.log_weights = torch.zeros(len(self.groups), dtype=torch.float64)

    def weights(self, device="cpu", dtype=torch.float32):
        return self.log_weights.softmax(0).to(device=device, dtype=dtype)

    def update(self, losses):
        values = torch.stack([losses[g].detach().cpu().double() for g in self.groups])
        self.log_weights += self.eta * values
        self.log_weights -= torch.logsumexp(self.log_weights, 0)

    def state_dict(self):
        return {"groups": self.groups, "eta": self.eta, "log_weights": self.log_weights.tolist()}

    def load_state_dict(self, state):
        if state["groups"] != self.groups: raise ValueError("DRO group mismatch")
        self.eta = state["eta"]
        self.log_weights = torch.tensor(state["log_weights"], dtype=torch.float64)
