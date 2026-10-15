"""Persist a network-free HF+LoRA integration artifact with actual measurements."""
from pathlib import Path
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from prmshift.model import StepScorer, causal_text
from prmshift.generate import generate
from prmshift.data import question_id
from prmshift.io import write_json, write_jsonl

torch.set_num_threads(2)
torch.manual_seed(2026)
out = Path("results/verification/local_hf")
out.mkdir(parents=True, exist_ok=True)
vocabulary = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "Problem": 3, ":": 4,
              "2": 5, "+": 6, "?": 7, "Solution": 8, "=": 9, "4": 10, "5": 11, ".": 12}
raw = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
raw.pre_tokenizer = Whitespace()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
base_dir = out / "random_base"
base_config = GPT2Config(vocab_size=13, n_embd=16, n_layer=1, n_head=2, n_positions=256,
                         resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, bos_token_id=2, eos_token_id=2, pad_token_id=0)
base = GPT2LMHeadModel(base_config)
base.save_pretrained(base_dir)
tokenizer.save_pretrained(base_dir)
model = StepScorer({"kind": "hf", "name": str(base_dir), "max_length": 256,
                    "target_modules": ["c_attn"], "lora_rank": 2, "lora_alpha": 4}).train()
before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
texts = [causal_text("2+2?", [], "2+2=4."), causal_text("2+2?", [], "2+2=5.")]
logits = model.score_texts(texts)
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0.]))
loss.backward()
grads = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
assert any(v > 0 for n, v in grads.items() if "lora_" in n)
opt.step()
changes = {n: float((p.detach() - before[n]).abs().max()) for n, p in model.named_parameters() if p.requires_grad}
assert any(v > 0 for n, v in changes.items() if "lora_" in n)
model.eval()
expected = model.score_texts(texts).detach()
model.save(out / "checkpoint")
restored = StepScorer.load(out / "checkpoint").eval()
actual = restored.score_texts(texts).detach()
max_error = float((actual - expected).abs().max())
assert max_error < 1e-6
questions = out / "questions.jsonl"
write_jsonl(questions, [{"qid": question_id("2+2?"), "problem": "2+2?", "answer": "4", "split": "test"}])
generation = generate({"model": str(base_dir), "generator": "random_local_gpt2_mechanism",
    "questions": str(questions), "output": str(out / "candidates.jsonl"), "chat_template": False,
    "candidates_per_question": 2, "max_new_tokens": 8, "max_prompt_tokens": 128, "device": "cpu", "seed": 2026})
report = {"scope": "random_initialization_software_integration_only", "pretrained_model_downloaded": False,
    "backbone": "GPT2LMHeadModel", "backbone_configuration": base_config.to_dict(),
    "head": "Linear(hidden=16, outputs=1)", "lora_target": "c_attn", "lora_rank": 2,
    "step_ce_before_update": float(loss.detach()), "gradient_norms": grads, "parameter_max_changes": changes,
    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "total_parameters": sum(p.numel() for p in model.parameters()), "restored_logit_max_abs_difference": max_error,
    "restored_logits": actual.tolist(), "generation_probe": generation,
    "checks": {"nonzero_lora_gradient": True, "adapter_updated": True, "save_load_matches": True,
               "frozen_generation_executed": True},
    "not_tested": ["pretrained Qwen 1.5B training", "CUDA memory", "real mathematics accuracy", "PRISM"]}
write_json(out / "report.json", report)
print(report["checks"])
