# Pretrained-model workflow

Run commands from the repository root after installing `.[hf,plots,test]`. Configurations use relative paths. Resolve each model's `revision: main` to an immutable commit before generating candidates so all methods can share the same cache.

## Prepare data

The downloader pins PRM800K to commit `7ecc794703b2877f63226f2477a49b34f9b25163` and reads the actual Git LFS content. The converter preserves branch structure, masks neutral or unavailable labels, and follows only the chosen or human continuation.

```bash
python -m prmshift download-prm --output data/raw \
  --files phase1_train.jsonl phase2_train.jsonl math_test.jsonl
python -m prmshift download-benchmark --name math500 --output data/evaluation/math500.jsonl
python -m prmshift download-benchmark --name processbench --output data/evaluation/processbench.jsonl
python -m prmshift prepare-prm \
  --inputs data/raw/phase1_train.jsonl data/raw/phase2_train.jsonl \
  --blocked data/raw/math_test.jsonl data/evaluation/math500.jsonl data/evaluation/processbench.jsonl \
  --output data/prepared_full
python -m prmshift budget-prm --input data/prepared_full --output data/prepared \
  --step-budget 10000 --seed 2026
python -m prmshift audit-splits \
  --inputs data/prepared/questions_train.jsonl data/prepared/questions_validation.jsonl \
    data/evaluation/math500.jsonl data/evaluation/processbench.jsonl \
  --output data/prepared/split_audit.json
```

PRM800K's expanded training split overlaps some original MATH test questions. Protected evaluation questions are excluded before the remaining questions are partitioned by hash. Exact overlaps fail the split audit; near-duplicate candidates are exported for review.

The budget command uses reservoir sampling to retain 10,000 explicitly labeled steps. Its manifest records the number available, number retained, question count and content hash. All four objectives use the same budgeted export.

## Generate candidate caches

```bash
python scripts/generate_splits.py --split train
python scripts/generate_splits.py --split validation
```

The training generators are configured in `generate_qwen.yaml` and `generate_qwen_general.yaml`. Both belong to the Qwen family. The cache records model revision, generation parameters, question and configuration hashes, generated tokens, truncation and answer-parsing status. A changed configuration cannot silently append to the old cache.

## Train and validate

```bash
python scripts/run_baselines.py --stage train --seed 2026
python scripts/run_baselines.py --stage validation --seed 2026
```

`hf_1_5b.yaml` specifies a Qwen2.5-Math-1.5B scorer with LoRA rank 8, context length 2048 and one-prefix microbatches. Set `iterations` to a small number for an initial capacity run on the target GPU. The reports record tokens, elapsed time and CUDA peak memory. Hardware requirements depend on the model revision, software stack and sequence lengths.

Select temperature, learning rate, CE weight and DRO step size using validation data, then freeze the configuration. Repeat matched comparisons across seeds. Training-step equality does not imply equal compute: set objectives make additional forward passes, which are counted in the reports.

## Evaluate

```bash
python scripts/generate_splits.py --split test
python scripts/run_baselines.py --stage test --seed 2026
python -m prmshift evaluate-processbench \
  --checkpoint results/formal/seed2026/group_dro/checkpoint \
  --data data/evaluation/processbench.jsonl \
  --output results/formal/processbench.json --device cuda
```

The test configuration adds DeepSeek-Math-7B as a held-out generator and pool size 16 as a held-out pool size. ProcessBench evaluates first-error localization separately from final-answer selection. The scorer never receives the answer key, future steps or generator identity as an input.

A pool with no correct candidate remains in coverage and accuracy reporting. Unknown outcomes remain unknown. If any candidate exceeds the context limit, evaluation excludes the entire question–generator cache and records the exclusion, rather than improving a pool by dropping its long candidates.

## Checkpointing and Slurm

Set `resume` to a checkpoint directory to restore model, optimizer, random-number state and DRO weights. Data hashes and sampling/objective settings are checked before resuming. Evaluation checks the saved training question IDs even if the original data path is absent.

`scripts/train_slurm.sh` is a single-GPU template. Set `PRMSHIFT_PYTHON`, `PRMSHIFT_REPO`, and the cluster's account/partition before submitting it. The script does not provision a server or submit itself.
