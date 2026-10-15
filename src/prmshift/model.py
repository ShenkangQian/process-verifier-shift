from __future__ import annotations
from pathlib import Path
import json
import torch
from torch import nn


class SequenceTooLong(ValueError):
    pass


def causal_text(problem, prefix, step):
    # Only observable text. Never serialize an entire labelled data record.
    return "Problem: " + problem + "\nSolution:\n" + "\n".join([*prefix, step])


class TinyBackbone(nn.Module):
    def __init__(self, width=32, layers=1, heads=4, max_length=512):
        super().__init__()
        self.embedding = nn.Embedding(259, width)
        self.position = nn.Embedding(max_length, width)
        layer = nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, ids, mask):
        x = self.embedding(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        causal = torch.triu(torch.ones(ids.shape[1], ids.shape[1], device=ids.device,
                                      dtype=torch.bool), diagonal=1)
        return self.blocks(x, mask=causal, src_key_padding_mask=~mask.bool())


class StepScorer(nn.Module):
    """Causal prefix scorer, with a fixed min-logit solution aggregator.

    Tiny mode is a byte-tokenized causal transformer for mechanism tests, not a
    pretrained mathematics model. HF mode trains an actual LoRA adapter and a
    binary head. Prefixes are scored separately to guarantee identical semantics
    across tokenizer families; this deliberately trades throughput for clarity.
    """
    def __init__(self, config, backbone=None, tokenizer=None):
        super().__init__()
        self.config = dict(config)
        self.kind = config.get("kind", "tiny")
        self.max_length = config.get("max_length", 512)
        self.tokenizer = tokenizer
        self.token_count = 0
        if self.kind == "tiny":
            width = config.get("width", 32)
            self.backbone = TinyBackbone(width, config.get("layers", 1), config.get("heads", 4), self.max_length)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
            if backbone is None:
                dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                         "float16": torch.float16}[config.get("dtype", "float32")]
                backbone = AutoModelForCausalLM.from_pretrained(config["name"],
                    revision=config.get("revision", "main"), torch_dtype=dtype,
                    trust_remote_code=False)
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config["name"],
                revision=config.get("revision", "main"), trust_remote_code=False)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                raise ValueError("Tokenizer requires a pad/eos token")
            self.tokenizer.padding_side = "right"
            if config.get("lora", True):
                backbone = get_peft_model(backbone, LoraConfig(task_type=TaskType.CAUSAL_LM,
                    r=config.get("lora_rank", 8), lora_alpha=config.get("lora_alpha", 16),
                    lora_dropout=config.get("lora_dropout", 0.0),
                    target_modules=config.get("target_modules", ["q_proj", "v_proj"])))
            if config.get("gradient_checkpointing", False):
                backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                if hasattr(backbone, "enable_input_require_grads"):
                    backbone.enable_input_require_grads()
            self.backbone = backbone
            resolved_revision = getattr(backbone.config, "_commit_hash", None)
            if resolved_revision:
                self.config["revision"] = resolved_revision
            self.backbone.config.use_cache = False
            width = getattr(backbone.config, "hidden_size", None) or backbone.config.n_embd
        self.head = nn.Linear(width, 1)
        # Two-pass selection gradients require deterministic scores. Disable
        # module dropout explicitly, including HF backbone and adapter dropout.
        if config.get("deterministic_scoring", True):
            for module in self.modules():
                if isinstance(module, nn.Dropout): module.p = 0.0

    @property
    def device(self):
        return self.head.weight.device

    def encode(self, texts):
        if self.kind == "tiny":
            sequences = [[1] + [byte + 3 for byte in text.encode("utf-8")] + [2] for text in texts]
        else:
            sequences = self.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
            if any(not seq for seq in sequences):
                raise ValueError("Empty token sequence")
        if any(len(seq) > self.max_length for seq in sequences):
            raise SequenceTooLong(f"Prefix exceeds {self.max_length} tokens; no silent truncation")
        pad = 0 if self.kind == "tiny" else self.tokenizer.pad_token_id
        ids = torch.full((len(sequences), max(map(len, sequences))), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            ids[i, :len(seq)] = torch.tensor(seq, device=self.device)
            mask[i, :len(seq)] = 1
        self.token_count += int(mask.sum().item())
        return ids, mask

    def score_texts(self, texts):
        ids, mask = self.encode(texts)
        if self.kind == "tiny":
            hidden = self.backbone(ids, mask)
        else:
            # LoRA lives inside the decoder's linear layers. Call that decoder
            # directly to avoid materializing unused [batch,length,vocab] LM
            # logits and every intermediate hidden layer in a 1.5B PRM run.
            causal_lm = self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone
            hidden = causal_lm.base_model(input_ids=ids, attention_mask=mask, use_cache=False,
                                           return_dict=True).last_hidden_state
        last = hidden[torch.arange(ids.shape[0], device=self.device), mask.sum(1) - 1]
        return self.head(last.to(self.head.weight.dtype)).squeeze(-1)

    def score_steps(self, problem, steps):
        if not steps:
            raise ValueError("A candidate must contain at least one nonempty step")
        # Sequential microbatches bound activation memory for full HF runs.
        texts = [causal_text(problem, steps[:i], step) for i, step in enumerate(steps)]
        microbatch = self.config.get("prefix_microbatch", 4)
        return torch.cat([self.score_texts(texts[i:i + microbatch])
                          for i in range(0, len(texts), microbatch)])

    def score_solution(self, candidate):
        return self.score_steps(candidate["problem"], candidate["steps"]).min()

    def score_candidates(self, candidates, batch=64):
        """Min-logit scores for many candidates, batching every prefix together.

        Used only for gradient-free diagnostics and validation; the same
        causal_text inputs as score_steps, padded with the key padding mask.
        """
        texts, owners = [], []
        for index, candidate in enumerate(candidates):
            for i, step in enumerate(candidate["steps"]):
                texts.append(causal_text(candidate["problem"], candidate["steps"][:i], step))
                owners.append(index)
        logits = torch.cat([self.score_texts(texts[i:i + batch]) for i in range(0, len(texts), batch)])
        scores = torch.full((len(candidates),), float("inf"), device=logits.device, dtype=logits.dtype)
        return scores.scatter_reduce(0, torch.tensor(owners, device=logits.device), logits, reduce="amin")

    def selection_backward(self, candidates, temperature, coefficient, return_scores=False):
        """Exact chain-rule subgradient of min-logit soft selection, two passes.

        First find min steps without graphs, then backpropagate through just the
        chosen min prefix of one candidate at a time. For ties choose the first
        minimizer, a valid subgradient. No straight-through approximation is used.
        """
        if not self.config.get("deterministic_scoring", True):
            raise ValueError("Memory-efficient selection requires deterministic scoring")
        from .objectives import soft_selection_loss
        minima, indices = [], []
        with torch.no_grad():
            for candidate in candidates:
                logits = self.score_steps(candidate["problem"], candidate["steps"])
                index = int(logits.argmin())
                indices.append(index)
                minima.append(logits[index])
            scores = torch.stack(minima)
            labels = [c["label"] for c in candidates]
            loss = soft_selection_loss(scores, labels, temperature)
            wrong = torch.tensor([not x for x in labels], device=scores.device, dtype=scores.dtype)
            derivatives = scores.div(temperature).softmax(0) * (wrong - loss) / temperature
        for candidate, index, derivative in zip(candidates, indices, derivatives):
            if derivative == 0: continue
            text = causal_text(candidate["problem"], candidate["steps"][:index], candidate["steps"][index])
            (coefficient * derivative * self.score_texts([text])[0]).backward()
        if return_scores:
            return loss, scores
        return loss

    def save(self, directory):
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model_config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        if self.kind == "tiny":
            torch.save(self.state_dict(), out / "weights.pt")
        else:
            self.backbone.save_pretrained(out / "adapter")
            self.tokenizer.save_pretrained(out / "tokenizer")
            torch.save(self.head.state_dict(), out / "head.pt")

    @classmethod
    def load(cls, directory, device="cpu"):
        out = Path(directory)
        config = json.loads((out / "model_config.json").read_text())
        if config.get("kind", "tiny") == "tiny":
            model = cls(config)
            model.load_state_dict(torch.load(out / "weights.pt", map_location="cpu", weights_only=True))
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                     "float16": torch.float16}[config.get("dtype", "float32")]
            base = AutoModelForCausalLM.from_pretrained(config["name"], revision=config.get("revision", "main"),
                                                       torch_dtype=dtype, trust_remote_code=False)
            backbone = PeftModel.from_pretrained(base, out / "adapter", is_trainable=True) if config.get("lora", True) else AutoModelForCausalLM.from_pretrained(out / "adapter")
            # Avoid applying a second adapter during construction.
            no_adapter = dict(config, lora=False, gradient_checkpointing=False)
            model = cls(no_adapter, backbone=backbone,
                        tokenizer=AutoTokenizer.from_pretrained(out / "tokenizer"))
            model.config = config
            model.head.load_state_dict(torch.load(out / "head.pt", map_location="cpu", weights_only=True))
        return model.to(device)
