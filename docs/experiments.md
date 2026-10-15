# Experiments

[Project overview](../README.md) · [Research design](research-design.md) · [Pretrained-model workflow](pretrained-workflow.md)

## Controlled cue experiment

`make_synthetic_cue` constructs arithmetic questions of the form “What is a+b+c?” with single-digit operands. Each seed has 300 training, 60 validation and 200 test questions. Every question–generator pair has 16 candidates in one fixed random order; smaller pools take prefixes of that order.

The first step can carry the suffix ` (verified)`. For a generator with direction `s` and strength `kappa`, the suffix probability is `0.5 + s*kappa` for a correct final answer and `0.5 - s*kappa` for an incorrect one. The actual arithmetic determines step labels independently. Traces include early errors, later errors and repaired errors.

| Generator | Used for training | Direction | Strength |
|---|---|---:|---:|
| `gen_A` | Yes | +1 | 0.35 |
| `gen_B` | Yes | +1 | 0.20 |
| `gen_C_heldout` | No | +1 | 0.30 |
| `gen_D_flipped` | No | −1 | 0.35 |

Step supervision has 1,800 labels per seed and exactly equal counts in the four cue × correctness cells. Consequently the cue itself has no label association for the step-only objective. The reversal experiment tests transfer of the correlation learned from candidate-set supervision.

The scorer is a causal byte-level transformer with width 32, one layer and four attention heads. The configuration uses AdamW, learning rate 0.003, step batch size 16, temperature 1.0, CE weight 1.0 and DRO step size 0.2. Each objective trains for 2,000 iterations. [`configs/cue_cpu.yaml`](../configs/cue_cpu.yaml) contains the parameters.

```bash
python scripts/run_cue.py --seeds 2026 2027 2028 2029 2030 \
  --iterations 2000 --validate-every 20 --test-questions 200 \
  --bootstrap 300 --threads 4 --parallel 3
```

The command writes fresh results to `results/cue_comparison/`, leaving the included results in `results/cue-study/` separate. It produces per-seed checkpoints, training reports, scored candidates and pool outcomes, then aggregates them into tables and figures.

For the additional temperature comparison recorded for seed 2026:

```bash
python scripts/run_cue.py --seeds 2026 --skip-existing --tau-ablation 0.1 0.3
```

To rebuild tables and figures from an existing complete run:

```bash
python scripts/run_cue.py --aggregate-only
```

## Metrics

For a pool with at least one correct candidate, conditional selection failure measures whether the highest-scoring candidate is incorrect. Coverage measures whether a correct candidate exists. Overall answer accuracy combines generation and selection. With known labels, accuracy equals coverage times one minus conditional failure.

Unknown labels yield lower and upper identification bounds. The plotted values are their midpoints. Intervals resample `(seed, question)` clusters jointly across generator and pool-size groups. They retain dependence between nested pools; they do not estimate model-seed uncertainty separately from question uncertainty.

The aggregate files include conditional failure, unconditional selection risk, coverage and overall accuracy for all groups. [`scorecard_numbers.json`](../results/cue-study/tables/scorecard_numbers.json) also records tests of the soft/hard selection relation and worst-group ranking. In this run, the predicted group-level ranking reversal and the predicted monotonic training-curve shape were not observed. The finite-pool inequality `hard_error <= N * soft_loss` held on all 180,000 logged batch pools.

## Reading the outputs

The included figures open at full resolution:

| Figure | What to read |
|---|---|
| [Conditional failure](../results/cue-study/figures/fig_a_conditional_failure_by_generator_and_n.png) | Each generator at each pool size |
| [Soft loss and hard error](../results/cue-study/figures/fig_b_soft_vs_hard_training_curves.png) | Differentiable objective and argmax error over training |
| [Selection margins](../results/cue-study/figures/fig_c_margin_histograms.png) | Score gaps for correct and incorrect top-1 selections |
| [Worst-group comparison](../results/cue-study/figures/fig_d_worst_group_table.png) | Worst groups under different risk definitions; also available as [searchable text](../results/cue-study/tables/worst_groups.txt) |

The [metrics table](../results/cue-study/tables/metrics.csv) and [aggregate JSON](../results/cue-study/aggregate.json) provide the corresponding numerical results. These links point to the recorded study; commands above write new outputs under `results/cue_comparison/`.

The experiment studies cue transfer in a controlled generator. Results for pretrained models should be reported in a separate table using their own cache manifests, model revisions and data splits.
