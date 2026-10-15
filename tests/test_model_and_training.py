from pathlib import Path
import copy
import torch
import pytest
from prmshift.model import StepScorer, TinyBackbone, causal_text
from prmshift.synthetic import make_synthetic
from prmshift.train import train
from prmshift.evaluate import evaluate
from prmshift.objectives import soft_selection_loss

torch.set_num_threads(2)


def test_tiny_causal_attention_and_metadata_noninput():
    torch.manual_seed(1)
    backbone = TinyBackbone(width=16, heads=2).eval()
    ids = torch.tensor([[1, 4, 5, 6]])
    changed = torch.tensor([[1, 4, 99, 98]])
    assert torch.allclose(backbone(ids, torch.ones_like(ids))[:, :2],
                          backbone(changed, torch.ones_like(ids))[:, :2], atol=1e-6)
    model = StepScorer({"kind": "tiny", "width": 16, "heads": 2, "max_length": 256}).eval()
    a = {"problem": "2+2?", "steps": ["2+2=4"], "label": True}
    b = dict(a, label=False, answer="secret", reward=100)
    assert torch.equal(model.score_solution(a), model.score_solution(b))


def test_two_pass_selection_gradient_equals_autograd():
    torch.manual_seed(7)
    a = StepScorer({"kind": "tiny", "width": 16, "heads": 2, "max_length": 256}).train()
    b = copy.deepcopy(a)
    cs = [{"problem": "2+2?", "steps": ["2+2=4.", "Answer: 4"], "label": True},
          {"problem": "2+2?", "steps": ["2+2=5.", "Answer: 5"], "label": False}]
    scores = torch.stack([a.score_solution(c) for c in cs])
    loss = soft_selection_loss(scores, [True, False], .7)
    loss.backward()
    two_pass = b.selection_backward(cs, .7, 1.)
    assert torch.allclose(loss, two_pass, atol=1e-6)
    for p, q in zip(a.parameters(), b.parameters()):
        if p.grad is not None: assert torch.allclose(p.grad, q.grad, atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize("method", ["ce", "balanced_ce", "set_erm", "group_dro"])
def test_actual_training_save_load_heldout(tmp_path, method):
    make_synthetic(tmp_path / "data", train_questions=6, test_questions=3)
    config = {"seed": 5, "method": method, "steps": str(tmp_path / "data/steps_train.jsonl"),
        "candidates": str(tmp_path / "data/candidates_train.jsonl"), "output": str(tmp_path / method),
        "iterations": 2, "step_batch_size": 3, "pool_sizes": [2], "learning_rate": .01,
        "model": {"kind": "tiny", "width": 16, "heads": 2, "max_length": 256}}
    report = train(config)
    assert report["history"][0]["gradient_norm"] > 0
    assert report["cost"]["optimizer_steps_this_run"] == 2
    model = StepScorer.load(config["output"]).eval()
    text = [causal_text("2+2?", [], "2+2=4.")]
    reference = model.score_texts(text).detach()
    model.save(tmp_path / "copy")
    restored = StepScorer.load(tmp_path / "copy").eval()
    assert torch.equal(reference, restored.score_texts(text).detach())
    result = evaluate(config["output"], tmp_path / "data/candidates_test.jsonl", tmp_path / "evaluation", [2], bootstrap=5)
    assert result["unique_questions"] == 3
    assert result["scope"] == ["synthetic_arithmetic_mechanism"]
    # Portability: moving raw steps away cannot disable leakage checks.
    from prmshift.io import read_jsonl, write_jsonl
    Path(config["steps"]).rename(tmp_path / "moved_steps.jsonl")
    leakage = [dict(r, split="test") for r in read_jsonl(config["candidates"])]
    write_jsonl(tmp_path / "leaked_test.jsonl", leakage)
    with pytest.raises(ValueError, match="Question leakage"):
        evaluate(config["output"], tmp_path / "leaked_test.jsonl", tmp_path / "must_not_run", [2], bootstrap=0)


def test_resume_matches_uninterrupted(tmp_path):
    make_synthetic(tmp_path / "data", train_questions=5, test_questions=2)
    base = {"seed": 17, "method": "group_dro", "steps": str(tmp_path / "data/steps_train.jsonl"),
        "candidates": str(tmp_path / "data/candidates_train.jsonl"), "iterations": 2,
        "step_batch_size": 2, "pool_sizes": [2], "model": {"kind": "tiny", "width": 16, "heads": 2, "max_length": 256}}
    train(dict(base, output=str(tmp_path / "full")))
    train(dict(base, iterations=1, output=str(tmp_path / "part")))
    train(dict(base, resume=str(tmp_path / "part"), output=str(tmp_path / "resumed")))
    a, b = StepScorer.load(tmp_path / "full"), StepScorer.load(tmp_path / "resumed")
    for p, q in zip(a.parameters(), b.parameters()): assert torch.equal(p, q)


def test_local_random_hf_lora_training_checkpoint(tmp_path):
    """Real HF+PEFT integration, no network, no pretrained capability claim."""
    pytest.importorskip("peft")
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tok = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "Problem": 3,
        ":": 4, "2": 5, "+": 6, "?": 7, "Solution": 8, "=": 9, "4": 10, "5": 11, ".": 12}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    base_path = tmp_path / "random_hf_base"
    backbone = GPT2LMHeadModel(GPT2Config(vocab_size=13, n_embd=16, n_layer=1, n_head=2,
                                        n_positions=128, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
    backbone.save_pretrained(base_path)
    tokenizer.save_pretrained(base_path)
    model = StepScorer({"kind": "hf", "name": str(base_path), "max_length": 128,
                        "target_modules": ["c_attn"], "lora_rank": 2, "lora_alpha": 4}).train()
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    texts = [causal_text("2+2?", [], "2+2=4."), causal_text("2+2?", [], "2+2=5.")]
    logits = model.score_texts(texts)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0.]))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_" in n)
    opt.step()
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if "lora_" in n and n in before)
    model.eval()
    expected = model.score_texts(texts).detach()
    model.save(tmp_path / "checkpoint")
    actual = StepScorer.load(tmp_path / "checkpoint").eval().score_texts(texts).detach()
    assert torch.allclose(expected, actual, atol=1e-6)
