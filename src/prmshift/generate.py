from __future__ import annotations
from pathlib import Path
import hashlib
import json
import re
import time
import torch
from .grading import extract_answer, grade_answer
from .io import read_jsonl, write_json, sha256, stable_seed


def split_steps(text):
    # Freeze this rule across methods; no model-dependent or label-dependent split.
    return [s.strip() for s in re.split(r"\n\s*\n", text.strip()) if s.strip()] or [text.strip() or "[empty response]"]


def generate(config):
    """Frozen HF generation with resumable, fingerprinted raw candidate cache."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA required by generation config")
    out = Path(config["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out.with_suffix(".manifest.json")
    fingerprint_data = {k: v for k, v in config.items() if k not in ("output", "device")}
    fingerprint_data["questions_sha256"] = sha256(config["questions"])
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["fingerprint"] != fingerprint:
            raise ValueError("Existing cache belongs to different model/questions/generation settings; choose new output")
    elif out.exists():
        raise ValueError("Existing cache without provenance manifest; refusing to append")
    write_json(manifest_path, {"fingerprint": fingerprint, "configuration": fingerprint_data,
                              "status": "generating", "generator_frozen": True})
    done = {(r["qid"], r["generator"], str(r["candidate_id"])) for r in read_jsonl(out)} if out.exists() else set()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[config.get("dtype", "float32")]
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config.get("revision", "main"), trust_remote_code=False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config.get("revision", "main"),
                                               torch_dtype=dtype, trust_remote_code=False).to(device).eval()
    model.requires_grad_(False)
    model_revision = getattr(model.config, "_commit_hash", None) or config.get("revision", "main")
    name = config.get("generator", config["model"])
    start = time.perf_counter()
    generated, prompt_tokens, output_tokens = 0, 0, 0
    with out.open("a", encoding="utf-8") as handle, torch.inference_mode():
        for q in read_jsonl(config["questions"]):
            prompt = "Solve the following mathematics problem step by step. Separate reasoning steps with a blank line and put the final answer in \\boxed{}.\n\n" + q["problem"]
            if config.get("chat_template", True) and tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            if inputs.input_ids.shape[1] > config.get("max_prompt_tokens", 1024):
                raise ValueError(f"Question {q['qid']} exceeds prompt limit; predefine a documented subset")
            for index in range(config.get("candidates_per_question", 16)):
                cid = f"{index:04d}"
                if (q["qid"], name, cid) in done: continue
                seed = stable_seed(config.get("seed", 2026), q["qid"], name, index)
                set_seed(seed)
                response = model.generate(**inputs, do_sample=True, temperature=config.get("temperature", .8),
                    top_p=config.get("top_p", .95), max_new_tokens=config.get("max_new_tokens", 1024),
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                new = response[0, inputs.input_ids.shape[1]:]
                text = tokenizer.decode(new, skip_special_tokens=True)
                answer = extract_answer(text)
                grade = grade_answer(answer, q.get("answer"))
                row = {"qid": q["qid"], "problem": q["problem"], "split": q["split"], "generator": name,
                       "candidate_id": cid, "steps": split_steps(text), "raw_response": text,
                       "extracted_answer": answer, "label": grade["label"], "grading_reason": grade["reason"],
                       "source": "frozen_hf_generator", "model_revision": model_revision, "generation_seed": seed,
                       "prompt_tokens": int(inputs.input_ids.shape[1]), "generated_tokens": int(len(new)),
                       "hit_token_limit": len(new) >= config.get("max_new_tokens", 1024)}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                generated += 1
                prompt_tokens += row["prompt_tokens"]
                output_tokens += row["generated_tokens"]
    manifest = {"fingerprint": fingerprint, "configuration": fingerprint_data, "status": "complete",
                "generator_frozen": True, "resolved_model_revision": model_revision, "cache_sha256": sha256(out),
                "this_invocation": {"new_candidates": generated, "prompt_tokens": prompt_tokens,
                                    "generated_tokens": output_tokens, "wall_seconds": time.perf_counter() - start},
                "cost_note": "Total cache generation tokens can be recomputed from every row; resumed invocation counts are incremental."}
    write_json(manifest_path, manifest)
    return manifest
