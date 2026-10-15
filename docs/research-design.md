# Research design: learning to choose among reasoning traces

[Project overview](../README.md) · [Experiment settings and figures](experiments.md) · [Pretrained-model workflow](pretrained-workflow.md)

The experiment separates three objects that are often compressed into one score: the correctness of a reasoning step, the availability of a correct candidate, and the verifier's decision once candidates are available. Each has a different supervision source and denominator. The implementation preserves those distinctions from data conversion through evaluation.

## The statistical target

Write a candidate pool as $S=\{r_1,\ldots,r_N\}$ for question $x$, with final-answer labels $z_i\in\{0,1\}$. Define $A=\{\sum_i z_i>0\}$, the event that the pool contains a correct candidate. A deterministic verifier selects an index $\hat i$ by score.

For generator–pool-size group $g$, let

$$
C_g=\Pr(A\mid g),\qquad
R_g^{\mathrm{hard}}=\Pr(z_{\hat i}=0\mid A,g).
$$

Because a selected correct answer implies $A$, answer accuracy factors as

$$
\Pr(z_{\hat i}=1\mid g)=C_g(1-R_g^{\mathrm{hard}}).
$$

This identity distinguishes a generation problem from a selection problem. Suppose two constructed groups have coverage and conditional failure $(0.9,0.2)$ and $(0.3,0.4)$. The second has worse conditional selection but lower unconditional selection-failure probability: $0.3\times0.4=0.12$ versus $0.9\times0.2=0.18$. Overall accuracies are 0.18 and 0.72. All three quantities describe different aspects of the same decision.

Training therefore conditions its selection loss on $A$, while evaluation retains all-incorrect pools in coverage and accuracy. The relationship is implemented in [`aggregate`](../src/prmshift/evaluate.py) and checked on explicit pools in [`test_coverage_identity_complete_labels`](../tests/test_objectives_and_metrics.py). It is a probability identity, independent of the model architecture.

## A differentiable objective with fixed scoring semantics

The input at step $t$ is `Problem + steps through t`. [`causal_text`](../src/prmshift/model.py) constructs that input explicitly rather than serializing a labeled record. Reference answers, final correctness, generator IDs and future steps cannot enter through metadata. The tiny backbone also applies causal attention; the test changes later tokens and checks that earlier hidden states stay unchanged.

Each candidate receives $s_i=\min_t h_\theta(x,r_{i,1:t})$. This treats one low-scoring step as sufficient to reduce the candidate's rank. Keeping this aggregation fixed makes the training-objective comparison interpretable. It also gives a concrete object to differentiate: one active prefix per candidate, with deterministic handling of tied minima.

With temperature $\tau>0$, define $p_i=\mathrm{softmax}(s/\tau)_i$ and

$$
\ell_\tau(S)=\sum_i (1-z_i)p_i.
$$

This loss is the error probability of sampling a candidate according to $p$. It differs from the deployed argmax error $e=\mathbf{1}\{z_{\hat i}=0\}$. A simple finite-pool relation explains why both are logged:

$$
e\leq N\ell_\tau.
$$

If the argmax is incorrect, its softmax mass is at least $1/N$ and contributes to the loss. Otherwise the left side is zero. The factor $N$ means a small soft loss must be interpreted relative to pool size. Lowering the loss can also leave the argmax unchanged. The experiment records soft loss and hard error on the same pools; it does not infer selection improvement from the surrogate alone.

## What group reweighting optimizes

Let $G$ contain the observed generator × pool-size groups and $R_g=\mathbb{E}[\ell_\tau\mid A,g]$. Set ERM assigns $q_g=1/|G|$, giving equal weight to groups rather than weighting every eligible pool equally. Group DRO uses an adversarial distribution $q$ to put more weight on larger observed risks:

$$
\min_\theta\;\lambda L_{\mathrm{step}}(\theta)
+\max_{q\in\Delta_G}\sum_g q_g R_g(\theta).
$$

The stochastic implementation samples one eligible question per group. At iteration $k$, model gradients use $q^{(k)}$ and sampled losses $\hat R_g(\theta^{(k)})$. After the optimizer step, detached versions of those losses update the weights:

$$
q_g^{(k+1)}=
\frac{q_g^{(k)}\exp\{\eta\hat R_g(\theta^{(k)})\}}
{\sum_h q_h^{(k)}\exp\{\eta\hat R_h(\theta^{(k)})\}}.
$$

[`GroupDRO`](../src/prmshift/objectives.py) stores normalized log weights; [`train`](../src/prmshift/train.py) implements this ordering. The method optimizes over observed groups. Any convex combination of their conditional risks is at most their maximum, but that algebra does not bound the risk of a new group with a changed cue–correctness relationship. The reversed-cue test probes precisely that distinction.

## Deriving the two-pass backward computation

Write $a_i=1-z_i$. Differentiating the softmax gives

$$
\frac{\partial p_j}{\partial s_i}
=\frac{p_j}{\tau}(\mathbf{1}\{i=j\}-p_i),\qquad
\frac{\partial\ell_\tau}{\partial s_i}
=\frac{p_i}{\tau}(a_i-\ell_\tau).
$$

For a selected minimizing step $t_i^{\ast}$, the corresponding parameter gradient is

$$
\nabla_\theta\ell_\tau=
\sum_i\frac{p_i}{\tau}(a_i-\ell_\tau)
\nabla_\theta h_\theta(x,r_{i,1:t_i^{\ast}}).
$$

This expression permits a graph-free pass to identify minima and compute coefficients, followed by one prefix graph at a time. No parameter update occurs between the two passes. Dropout is disabled, so recomputation uses the same scoring function. Tied minima select the first minimizing step; this chooses a valid subgradient rather than averaging across tied branches.

[`selection_backward`](../src/prmshift/model.py) accumulates each contribution directly into parameter gradients. [`test_two_pass_selection_gradient_equals_autograd`](../tests/test_model_and_training.py) checks loss agreement and parameter-gradient agreement against the full graph. The optimization trades additional forward work for fewer simultaneously retained activations. Token accounting includes the repeated prefixes, while diagnostic passes have a separate counter.

## Designing a shift that can be inspected

The synthetic generator defines local step correctness by the arithmetic written in that step. Final correctness is computed independently from the problem's answer. Thus a repaired trajectory can contain an incorrect first step and a correct final answer; an early error can propagate through a locally correct second calculation.

The stylistic cue is the suffix ` (verified)`. For sign $s$ and strength $\kappa$, its probabilities are

$$
\Pr(\mathrm{cue}\mid z=1,g)=0.5+s\kappa,\qquad
\Pr(\mathrm{cue}\mid z=0,g)=0.5-s\kappa.
$$

Training uses positive associations of strengths 0.35 and 0.20. The held-out same-direction generator uses 0.30; the reversed generator uses $s=-1$ and strength 0.35. These are generator distributions with controlled association parameters, not paired copies of identical traces. The manifest records realized contingencies, coverage and trajectory types so the realized sample can be checked against the specified probabilities.

Step supervision balances cue presence across correct and incorrect labels. Candidate-set supervision, in contrast, can exploit the cue's association with final correctness. This design asks whether the learned selection rule transfers when that association changes. It does not assume that a real mathematical generator has the same cue structure.

All methods share the candidate cache and initialization within a seed. Nested pool sizes use prefixes of one fixed random permutation for each question–generator pair. Candidate counts therefore change the selection problem without introducing a second independently sampled dataset. [`test_cue_experiment.py`](../tests/test_cue_experiment.py) verifies cue contingencies, local arithmetic labels, split behavior and nested membership.

The recorded comparison shows lower conditional failure for selection-trained scorers on the same-direction held-out generator, and higher failure on the reversed generator. Group DRO follows the same reversal. The interpretation concerns which supervision signal the scorer learned to use; the complete group table remains available rather than reducing the result to one pooled accuracy.

## Unknown labels, dependence and reproducibility

Training removes unknown final labels before forming pools and records their effective size. Evaluation keeps unknown candidates. For each pool it records definitely or possibly covered status and definitely or possibly failed selection. Let their aggregate counts be $C_L,C_U,F_L,F_U$. When $C_L>0$, a conservative interval for conditional failure is

$$
\left[\frac{F_L}{C_U},\;\min\left(1,\frac{F_U}{C_L}\right)\right].
$$

[`aggregate`](../src/prmshift/evaluate.py) handles zero-denominator cases explicitly. The bounds express label uncertainty; bootstrap intervals add sampling uncertainty. The cue-study bootstrap resamples `(seed, question)` clusters with shared multiplicities across groups and nested pool sizes. It does not separately estimate model-seed variance.

The implementation also treats reproducibility as a data-integrity problem. Question hashes preserve split boundaries, candidate caches record inputs and revisions, and checkpoints store training question IDs, optimizer state, RNG state and DRO weights. Tests verify uninterrupted-versus-resumed parameter equality and reject held-out evaluation on training questions even after original files move. [Experiment settings](experiments.md) and [the pretrained-model workflow](pretrained-workflow.md) provide execution instructions for these interfaces.
