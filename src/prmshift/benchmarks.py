from pathlib import Path
from .data import question_id
from .io import write_jsonl, write_json


def download_benchmark(name, output, revision="main", max_records=None):
    """Download public held-out sets. Pin and report resolved dataset revision.

    MATH500 questions also serve as protected exclusions before PRM preparation.
    ProcessBench carries first-error labels, not final answer correctness.
    """
    from datasets import load_dataset
    specs = {"math500": ("HuggingFaceH4/MATH-500", None, "test"),
             "gsm8k": ("openai/gsm8k", "main", "test"),
             "processbench": ("Qwen/ProcessBench", None, "test")}
    if name not in specs: raise ValueError(f"Supported benchmarks: {list(specs)}")
    repository, subset, split = specs[name]
    if name == "processbench":
        # Public repository exposes four independent benchmark files.
        datasets = {part: load_dataset(repository, data_files=f"{part}.json", split="train", revision=revision)
                    for part in ("gsm8k", "math", "olympiadbench", "omnimath")}
    else:
        datasets = {name: load_dataset(repository, subset, split=split, revision=revision)}
    rows = []
    fingerprints = {}
    for part, dataset in datasets.items():
        fingerprints[part] = dataset._fingerprint
        for row in dataset:
            problem = row.get("problem") or row.get("question")
            if not problem: raise ValueError(f"Unsupported {name} schema; missing problem")
            result = {"qid": question_id(problem), "problem": problem, "split": "test", "source": f"{repository}/{part}"}
            if name == "processbench":
                result.update({"steps": row["steps"], "first_error": row["label"], "subset": part})
            else:
                answer = row.get("answer")
                if name == "gsm8k": answer = answer.rsplit("####", 1)[-1].strip()
                result["answer"] = answer
            rows.append(result)
            if max_records and len(rows) >= max_records: break
        if max_records and len(rows) >= max_records: break
    write_jsonl(output, rows)
    write_json(Path(output).with_suffix(".manifest.json"), {"repository": repository, "requested_revision": revision,
        "dataset_fingerprints": fingerprints, "records": len(rows), "partial_sample": max_records is not None})
    return {"records": len(rows), "repository": repository}


def evaluate_processbench(checkpoint, path, output, device="cpu", threshold=0.0):
    import torch
    from .model import StepScorer, SequenceTooLong
    from .io import read_jsonl
    model = StepScorer.load(checkpoint, device).eval()
    rows, excluded = [], 0
    with torch.inference_mode():
        for row in read_jsonl(path):
            try: scores = model.score_steps(row["problem"], row["steps"]).tolist()
            except SequenceTooLong:
                excluded += 1
                continue
            pred = next((i for i, value in enumerate(scores) if value < threshold), -1)
            rows.append({"qid": row["qid"], "prediction": pred, "label": row["first_error"], "subset": row["subset"]})
    report = {}
    for subset in sorted({r["subset"] for r in rows}):
        part = [r for r in rows if r["subset"] == subset]
        errors = [r for r in part if r["label"] != -1]
        clean = [r for r in part if r["label"] == -1]
        a = sum(r["prediction"] == r["label"] for r in errors) / len(errors) if errors else None
        b = sum(r["prediction"] == -1 for r in clean) / len(clean) if clean else None
        report[subset] = {"questions": len(part), "first_error_accuracy_on_erroneous": a,
                          "correct_solution_accuracy": b,
                          "harmonic_mean": None if a is None or b is None else (2*a*b/(a+b) if a+b else 0.0)}
    write_json(output, {"subsets": report, "threshold": threshold, "context_excluded": excluded,
                        "note": "Threshold must be frozen using separate validation questions; no test tuning."})
    return report
