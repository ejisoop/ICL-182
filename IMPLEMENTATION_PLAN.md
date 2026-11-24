# Implementation Plan

Project: *Can One Attention Layer Help a Recurrent Model Learn Rules from Examples?*
Source of truth: `RESEARCH_PROPOSAL.md`. This plan was written before any main-experiment results existed. The analysis plan in §6 is pre-registered and must not change after results are seen. Any later deviation is recorded in `EXPERIMENT_LOG.md` and the final report.

## 1. Environment (inspected 2026-11-18)

| Item | Value |
|---|---|
| Hardware | 7× NVIDIA L40S (46 GB) on a shared lab server; 128 CPU cores; 377 GB RAM |
| GPU policy | user `eric` may run **at most 2 GPU processes in total** (all sessions/projects). This is enforced by `scripts/run_experiments.py`, which counts every process eric has on the GPUs. |
| Python | 3.11 (uv venv at `.venv`) |
| PyTorch | 2.7.1+cu126, Triton 3.3.1 |
| Flash Linear Attention | `flash-linear-attention==0.6.0`. `fla.layers.GatedDeltaNet` works on L40S (forward and backward smoke test passed). |
| Existing repository | Only `CLAUDE_CODE_PROMPT.md` and `RESEARCH_PROPOSAL.md`. There was no prior code. |

## 2. Task design (`src/data/`)

* **Language.** Every token is a word. Assignments are `v = n`. Built-in updates are `set|add|sub|mul v arg`, where `arg` is a number or a variable. Made-up commands are `cmd v arg`, with `cmd ∈ {dax, wug, blick, fep, zup, kiki, toma, lorp}`. Variables are `a`–`h`. All arithmetic is **mod 13**, so every value and answer is a single token in `0..12`.
* **Demonstration.** Format: `D v = x ; cmd v n ; v -> result .`. Each demonstration is *individually unambiguous*: the result under the true operation differs from the result under each of the other three operations.
* **Prompt.**
  * Demonstrations: `<bos>`, then the interleaved demonstrations for 3 made-up commands (1 target and 2 distractors, k demonstrations each), then `Q <query program> <var> ->`.
  * Direct-definition control: `<bos> M cmd means op .` for the same 3 commands, then the same query.
* **Query program (fixed complexity).** *Final (after calibration, see §5 and EXPERIMENT_LOG):* **two initialisations followed by 3 updates** (`n_init=2`, `n_stmts=3`). Exactly one update uses the target command; the others are built-ins. *(The original plan used 3 inits + 4 updates. That proved to be at the floor in pilot v1 and was changed before any main run.)*
* **Paired case.**
  * The two prompts share the query program, initial values, query variable, distractor commands, their meanings and their demonstrations.
  * The target meanings differ (A ≠ B). Only the target command's demonstrations differ between the two prompts.
  * The query variable is chosen so that its final value differs between A and B.
  * Demonstrations are nested across k: the prompt for k uses a subset of the demonstrations for k_max = 8.
  * Consequently, at k = 0 the two prompts are *identical text with different answers*, so paired accuracy is exactly 0 by construction. This serves as a sanity floor.
* **Interpreter** (`interpreter.py`). It parses rendered text and is independent of the generator. All ground-truth answers come from it.
* **Splits / leakage.**
  * A query-program *template* is (target position, built-in ops, data flow = which variable slot each statement updates and reads/argument type). For the final 2×3 task there are 10,368 templates, split 8,376 / 1,010 / 982 (train / val / test).
  * Templates are hashed deterministically into train (80%), val (10%) and test (10%).
  * Training, validation and test therefore use disjoint program structures. Exact-program leakage is tested as well.
* **Training data.**
  * Generated on the fly. Example *i* of seed *s* is a deterministic function of (*s*, *i*), so the stream is identical across architectures for the same seed and resumption is exact.
  * Mix: 75% demonstration prompts with k ~ U{0..8} and 25% direct-definition prompts.
  * Every training prompt is one side of a freshly generated case from the train templates, so training and test distributions differ only in template structure.
  * Loss is applied at every interpreter-determined result position: the token after each `->` (demonstration results and the final answer).
* **Fixed evaluation data.**
  * `data/test_cases.jsonl`: 1000 paired cases from the test templates.
  * `data/val.jsonl`: 3000 prompts from the val templates.
  * Both are generated with fixed seeds by `scripts/generate_data.py`, and SHA-256 hashes are recorded in `data/data_manifest.json`.

## 3. Models (`src/models/`)

All models are pre-norm residual stacks of 8 blocks with d_model = 256. Each block applies a mixer and then a SwiGLU MLP (hidden size 704). The remaining components are the same in every model:
* RMSNorm.
* Untied embedding and LM head, with the same vocabulary of 44 tokens.

| Model | Mixers | Params |
|---|---|---|
| `pure` | 8 × `fla.layers.GatedDeltaNet` (4 heads × 64, expand_v = 1, short conv 4, output gate) | 7,015,232 |
| `hybrid_early` | GDN, except layer **2** = causal softmax MHA (4 heads × 64, RoPE) | 6,944,504 |
| `hybrid_middle` | attention at layer **4** | 6,944,504 |
| `hybrid_late` | attention at layer **6** | 6,944,504 |

* **Convention.** The hybrid replaces the *mixer* of one block with attention and keeps depth, MLP and norms unchanged.
* **Parameter gap.** The hybrids have 1.0% fewer parameters than the pure model, because one GDN mixer (333k params) is larger than one attention mixer (262k). The gap goes against the hypothesis, so it cannot explain a hybrid advantage. It is reported, not compensated.

## 4. Training (`src/training/`, `configs/base.yaml`)

* **Optimiser.** AdamW with lr 1e-3, betas (0.9, 0.95), weight decay 0.1 on matrices, and gradient clipping at 1.0.
* **Schedule and precision.** 1000 warmup steps, then cosine decay to 10%. Batch size 128. bf16 autocast.
* **Steps.** Identical for all architectures. The value is fixed after the pilot (§5).
* **Logging.** Train loss, answer loss, learning rate, gradient norm and elapsed time are logged every 50 steps. Validation answer loss and accuracy (overall, ICL and definition) are logged every 1000 steps.
* **Checkpoints and metadata.**
  * Checkpoints are saved every 2000 steps, including optimiser and RNG state. Training resumes automatically.
  * `metadata.json` stores the configuration, parameter breakdown, environment, hardware and duration.
* **Seeds.** 0–4 for every architecture, giving 4 × 5 = 20 runs. Seed *s* fixes both the initialisation and the training-data stream, so all architectures with seed *s* see identical data.
* **Checkpoint used for test.** The **final** checkpoint is evaluated. There is no early stopping and no checkpoint selection on test data.

## 5. Pilot (calibration only)

* **Runs.** `pure` and `hybrid_middle` with **seed 100**, which is not a main-experiment seed, for 10k steps.
* **Purpose.** Validate the pipeline end to end, measure throughput, and choose the step budget.
* **Budget criterion, decided before the pilot.** Choose the step budget so that 20 runs fit in the available GPU time and validation ICL accuracy of the pilot models has clearly left chance. The test set is not used for calibration. The budget is fixed for all architectures before any main run.

* **What actually happened (deviation, documented in EXPERIMENT_LOG).**
  * Pilot v1 (3×4 task) sat near the floor (≈21% validation accuracy for both architectures) and was stopped.
  * Two calibration rounds followed, both with **pure only** and validation data only, under rules stated before each round.
  * Round 2 (20k steps; {2×2, 2×3} × lr {1e-3, 2e-3}) selected **2×3, lr 1e-3, 20k steps**.
  * The ≥70% definition-accuracy target was not reached by any configuration (best 55%), so the fallback clause applied.

## 6. Pre-registered evaluation and analysis

**Evaluation.**
* All 20 models are evaluated on the identical 1000 test cases.
* ICL is evaluated with k ∈ {0, 1, 2, 3, 4, 6, 8} demonstrations per command. The control (definition) condition is also evaluated.
* Each case contributes 2 prompts per condition, for 16,000 prompts per model.
* The prediction is the argmax over the full vocabulary. A non-number counts as wrong.

**Metrics.**
* Paired accuracy: the fraction of pairs where both prompts are correct.
* Individual accuracy.
* Accuracy by operation, by target command, by k and by seed.
* Validity rate of the output format.

**Statistical unit.** The **training seed** (n = 5 per architecture) is the unit for all between-architecture inference. Test prompts are never treated as independent replicates of training runs. Example-level uncertainty is addressed separately by a hierarchical bootstrap that resamples seeds within each architecture and test cases shared across architectures.

**Primary endpoint (H1).** *ICL paired accuracy averaged over k ∈ {1, 2, 3, 4, 6, 8}*, compared for each hybrid against pure.
* Welch t-test on the 5 vs 5 per-seed values, with Holm correction across the 3 hybrid-vs-pure comparisons.
* Exact permutation test (252 relabelings) as a robustness check.
* Report the mean difference, 95% CI, Cohen's d and the hierarchical-bootstrap CI.
* Secondary: the same comparison at each k, and a small-demonstration endpoint (mean over k ∈ {1, 2}). These are reported descriptively with uncorrected p-values flagged as exploratory.

**H2 (placement).**
* One-way ANOVA and Kruskal–Wallis across early, middle and late on per-seed primary-endpoint values.
* Pairwise Welch tests with Holm correction.

**H3 (rule inference vs execution).**
* For each seed, compute d = (ICL primary endpoint) − (definition paired accuracy).
* The interaction (hybrid advantage in ICL minus hybrid advantage in control) equals mean_hybrid(d) − mean_pure(d).
* Test it with Welch and permutation tests for each placement, with Holm correction.
* Ceiling effects in the control will be checked and reported, because they can make the interaction uninterpretable.

**Significance.** α = 0.05 (two-sided). With n = 5 seeds, power is limited. Non-significant results will be reported as inconclusive, not as evidence of no effect.

## 7. Deliverables

* Code: `src/`, `scripts/` and `tests/` (pytest).
* Outputs: checkpoints in `checkpoints/main/`, raw per-prompt predictions in `results/raw/`, tables in `results/tables/`, figures (PNG and PDF) in `results/figures/`, and the report in `reports/FINAL_REPORT.md`.
* Documentation: `README.md` and `EXPERIMENT_LOG.md`.

## 8. Risks and mitigations

* **Ceiling or floor effects** would make all architectures look identical. The pilot checks for this. If both architectures saturate, this is reported honestly. Changing difficulty would be a documented protocol deviation made *before* the main runs.
* **GPU contention from other users or sessions.** The scheduler waits for the global cap, and jobs resume from checkpoints.
* **The `fla` Triton kernels are non-deterministic in their reductions**, so bit-exact reproducibility of training is not guaranteed. Evaluation given a checkpoint is deterministic.

## 9. Experiment 2: ambiguous demonstrations (pre-registered 2026-11-19, before any Experiment-2 run)

**Motivation (from Experiment 1).** In Experiment 1 one individually unambiguous demonstration already identified the meaning: ICL accuracy at k = 1 equalled the definition control in every model. The hypotheses about *rule inference* were therefore not really exercised. Experiment 2 makes inference necessary and keeps everything else fixed.

**Task change (only change).** `data.ambiguous_demos: true` (`configs/ambiguous.yaml`).
- Every demonstration (target and distractors) is consistent with **exactly two** operations: the true one and a "confuser" drawn uniformly from the other three. The (init, arg) pair is then drawn uniformly from those realising exactly that pair.
- One demonstration never identifies the meaning. k demonstrations identify it unless all share the same confuser, which happens with probability 3^(1−k): 0% identifiable at k = 1, ≈ 67% at k = 2, ≈ 96% at k = 4.
- Query programs (2 inits + 3 updates), templates and splits, model sizes, optimiser, 20k steps and seeds 0–4 are all unchanged.
- Fixed data: `data/amb/` (hashes in `data/amb/data_manifest.json`). There is no calibration: execution difficulty is unchanged from Experiment 1.

**Design.** 4 architectures × seeds 0–4 = 20 runs (`checkpoints/amb/`, `results/raw_amb/`). The final checkpoint is evaluated on 1000 paired cases × {k = 0,1,2,3,4,6,8, definition}.

**Analysis (`scripts/analyze_exp2.py`).** Endpoints are as in §6: ICL paired accuracy over k ∈ {1,2,3,4,6,8} (primary), k ∈ {1,2} (small-k), definition paired accuracy, and the ICL − definition interaction.

The **primary test is now the seed-paired one.** Experiment 1 showed that architectures sharing a seed share the outcome regime, so the seed is a blocking factor. This decision is made before any Experiment-2 data exist.
- **H1:** for each hybrid vs pure, a paired t-test on the 5 per-seed differences of the primary endpoint, Holm-corrected over the 3 hybrids. Also reported: an exact sign-flip permutation test (32 sign patterns), the mean difference with 95% CI, and, secondarily, the Experiment-1 unpaired tests (Welch, permutation, hierarchical bootstrap).
- **H1 small-k:** the same tests on k ∈ {1,2}, Holm-corrected separately.
- **H2:** repeated-measures comparison of the three placements. Friedman test, plus paired t-tests between placements with Holm correction.
- **H3:** the per-seed interaction (hybrid − pure on ICL) − (hybrid − pure on definition), tested with a paired t-test and sign-flip permutation, Holm over 3.
- **Exploratory:**
  - accuracy split by whether the shown demonstrations identify the meaning (via the interpreter's consistent-op set);
  - accuracy on non-identifiable prompts relative to the 50% Bayes ceiling for picking between 2 ops;
  - per-k differences.

**Expected pattern if H1 is true:** hybrid > pure in ICL at small and intermediate k, with a smaller or no difference in the definition control.

*Addendum, written before any Experiment-2 run.* With n = 5 seeds the exact sign-flip test has a minimum attainable two-sided p of 2/32 = 0.0625. It is therefore reported as supporting evidence only, and **the decision rule is the Holm-corrected paired t-test at α = 0.05**. `scripts/analyze_exp2.py` was dry-run on synthetic predictions in the scratchpad (not `results/`) to check that it recovers injected effects.

## 10. Post-hoc follow-ups (not pre-registered; pointers)

These were designed after seeing Experiment-1 results. Each design and its reading rule or prediction was written in `EXPERIMENT_LOG.md` *before* the runs. Results are reported as exploratory (report §10, §10.1).
- **Data vs init** (`configs/seedsplit_jobs.json`): pure model, data-stream seed × init seed. EXPERIMENT_LOG "Exploratory follow-up".
- **Experiment A, early-layer init transplant** (`configs/earlyinit_jobs.json`, `--early-init-seed`). EXPERIMENT_LOG Phase 7. The prediction failed.
- **Experiment 2 post-hoc additions:** the per-run take-off / k = 1 ceiling table and within-seed concordance (`exp2_runs.md`) were added after seeing Experiment-2 results and are labelled exploratory in the report (§11.2–11.3).

## 11. Load-scaling pilot (pre-registered 2026-11-21 00:05, before any pilot run)

**Question.** Is the direction viable at this scale at all? Does one attention layer help once the number of command bindings a prompt holds exceeds what a Gated DeltaNet state can store, with program execution made trivial so the take-off lottery of Experiments 1–2 cannot decide the outcome?

**Task** (`src/data/recall.py`, `configs/recall.yaml`).
- N distinct made-up commands from a pool of 512. Each means "add c" (mod 13), with c drawn fresh per prompt.
- Bindings use one format per prompt:
  - "icl": `D cmd x = y ;` (inference: c = y − x);
  - "definition": `M cmd c ;` (control).
- Then 8 queries `Q cmd x -> a ;` on distinct bound commands. Only query answers are supervised. Chance is 7.7%.
- Training: N uniform over {4, 8, 16, 32, 64, 128}, 25% definition prompts.
- A GDN head's key space is 64-dimensional, so N = 64–128 should exceed what one head can store without interference.

**Design.**
- Same models and optimiser as Experiments 1–2.
- Batch 64, 20k steps; the measured speed is ~0.5 s/step (smoke test, seed 99, not analysed).
- 4 architectures × seeds {0, 1} = 8 runs (`checkpoints/recall/`).
- Fixed test set: 250 prompts × 8 queries for every (N, format). That is 2000 queries per cell (`data/recall/test.jsonl`, sha256 `fe75dce5…`).

**Primary readout.** Query accuracy in the icl format at N = 128 (and N = 64), hybrid − pure, per seed (`scripts/analyze_recall.py`). With 2 seeds this is a descriptive pilot: no significance test, and the rules below are fixed now.

**Decision rules**, applied in order:
1. **Failed run:** a run whose icl accuracy at N = 4 is < 50% did not learn the task. It is reported as such, and differences involving it are not interpreted as capacity effects. If pure fails in both seeds and some hybrid learns in both, that is reported as a *learnability* separation.
2. **Load too low:** in both seeds, pure's icl accuracy at N = 128 is ≥ 90% of its N = 8 accuracy. The state is not stressed. Next step: extend N (≥ 256) before any conclusion.
3. **Separation (direction alive):** some hybrid beats pure by ≥ 10 pp at N = 128 (icl) in **both** seeds. Next step: a pre-registered confirmatory experiment with ≥ 5 seeds, Holm-corrected seed-paired tests and placement comparison.
4. **Inconclusive:** a hybrid beats pure by ≥ 10 pp at N = 128 in exactly one seed. Next step: add seeds before concluding.
5. **"Cooked" at this scale:** pure degrades with load (rule 2 not met), and no hybrid beats it by ≥ 10 pp at N = 128 in either seed. Conclusion: one attention layer does not rescue binding capacity here, and the direction is not worth pursuing at this model size.

**Secondary (descriptive):**
- the same tables for the definition format;
- the icl − definition gap per N (an H3 analogue);
- accuracy by position of the queried binding at N = 128 (recency).

## 12. MQAR load-scaling pilot (pre-registered 2026-11-23 PST, before sizing or training)

**Question.** Does replacing one Gated DeltaNet mixer with causal softmax attention improve associative recall under binding load?
The failed §11 pilot combined retrieval with modular arithmetic, which left both observed runs near chance even with explicit definitions.
MQAR makes the supervised answer the stored value itself and uses an all-attention transformer as a positive control.

### 12.1 Task, models and measurements

- Use the 512 existing command tokens as keys and 64 new value tokens `v0` through `v63`.
  Start each prompt with `<bos>` and emit `N` bindings `M k v ;` with distinct keys in random order and independent uniform values.
  Then sample eight distinct bound keys in random order and emit `Q k -> v ;` for each, using four distinct queries at `N = 4`.
  Supervise only the query answer tokens, with causal next-token cross-entropy over the full vocabulary.
  At evaluation, take the argmax over the 64 legal values, for a chance accuracy of 1/64 (1.5625%).
- Train with `N` uniform over `{4, 8, 16, 32, 64, 128, 256}` and a deterministic stream keyed by training-data seed and example index.
  Architectures with the same seed share data and use that seed for initialization.
  Freeze 100 validation prompts and 250 test prompts per `N`, from separate deterministic streams with split hashes in `data/mqar/data_manifest.json`.
  This yields 400 validation and 1,000 test queries at `N = 4`, and 800 validation and 2,000 test queries at each other load.
  Record zero-based binding rank from oldest to newest for every query.
- Use the same eight pre-norm residual blocks, `d_model = 256`, four mixer heads, SwiGLU, untied embeddings, and optimizer as in `configs/base.yaml`, with the MQAR vocabulary in all arms.
  In `transformer` every mixer uses the hybrids' existing causal SDPA and RoPE.
  The three hybrids replace one mixer at zero-based layer 2, 4 or 6.
  For `gdn_head_dim = 32`, reduce only the GDN query and key head width from 64 to 32, holding GDN value head width and attention head width at 64.
  Record parameter counts without scaling any other model component to compensate.

### 12.2 Sizing and validation-only learnability gates

1. First size one otherwise representative `hybrid_middle` run (`gdn_head_dim = 64`, seed 99, 400 steps, batch 64) using GNU `/usr/bin/time -v`.
   Record peak RSS, peak observed threads, CPU utilization, seconds per step and GPU memory headroom.
   Use `num_workers = 0`, one thread for OMP, MKL, OpenBLAS and Inductor, the scheduler with `--max-jobs 2`, mandatory headroom checks and `systemd-run --user --scope` hard memory and task caps.
   Use the initial 4 GiB and 128-task caps from Phase 8, with any sizing-driven operational adjustment logged before fan-out.
   Run on the authorized Thunder Compute L40, with one GPU available and a hard ceiling of two distinct GPUs across concurrent jobs.
   On the dedicated six-vCPU Thunder host, require at least 8 GiB available RAM, a one-minute load average below 4.5 and fewer than 2,048 tasks in the user slice.
2. Run `transformer`, seed 0, for at most 5,000 steps, with batch 64 and validation only.
   Abort at step 2,000 if the logged train answer loss over steps 1,951–2,000 is within 0.05 of `ln(64) = 4.158883`.
   Diagnose task generation, masking and the learning rate from the existing train and validation evidence if the abort fires.
   The gate passes if the step-5,000 validation query accuracy is at least 90% at `N = 16`.
   If the transformer gate fails, stop and report the failure before running anything else.
3. After the transformer passes, run `pure`, `gdn_head_dim = 64`, seed 0, for 5,000 steps with the same batch and optimizer schedule, using validation only.
   This run must reach at least 90% validation query accuracy at `N = 8` at step 5,000.
   If it fails, report it before launching the pilot.

### 12.3 Pilot arms and analysis

Launch the pilot only after both gates pass, with fresh initialization and a full 20,000-step schedule, batch 64 and the remaining `configs/base.yaml` optimizer settings.
Run two jobs at a time within resource headroom, with seeds `{0, 1}` in every arm, for 14 runs total.

| Arm | GDN key head width | Architectures | Runs |
|---|---:|---|---:|
| A | 64 | `transformer`, `pure`, `hybrid_early`, `hybrid_middle`, `hybrid_late` | 10 |
| B | 32 | `pure`, `hybrid_middle` | 4 |

Evaluate each run on the fixed test set only after it reaches the final checkpoint.
Keep one raw CSV row per query and report query accuracy per `N`, hybrid minus pure accuracy in percentage points by seed and matching head width, and accuracy by binding rank at `N = 128` and `N = 256`.
Show seeds individually in all comparisons and use 16 equal-width binding-rank bins for the recency figures, with exact rank scores also tabulated.
Placement ordering, the load curves, the recency profiles and the shift in pure-GDN degradation under Arm B are descriptive only.
Report the degradation point as the first tested `N > 8` below 90% of the same seed's `N = 8` accuracy, or record that this point was not reached.

### 12.4 Ordered decision rules

1. **R1, failed run.** Test query accuracy below 90% at `N = 8` is a learnability failure.
   Report each failure and exclude capacity interpretations involving that run.
2. **R2, broken control.** Transformer test accuracy below 90% at `N = 128` in either seed indicates a task or budget issue.
   Report it before interpreting capacity differences.
   Passing this threshold in both seeds is the positive-control requirement used by R6.
3. **R3, load too low.** If 64-dim pure at `N = 256` retains at least 90% of its `N = 8` accuracy in both seeds, use Arm B as the primary readout.
   Continue to R4 and R5 in Arm B and report R3 alone if neither qualifying gap occurs.
4. **R4, separation.** A hybrid beats pure by at least 10 percentage points at the same `N` (128 or 256) and matched GDN head width in both seeds.
   If R3 routed the decision to Arm B, use only Arm B for this decision.
   Propose a confirmatory experiment with at least five seed-paired runs per architecture, a Holm correction and a placement comparison, without launching it.
5. **R5, inconclusive.** If R4 is unmet, a matched comparison reaches the 10-point threshold at exactly one of the two seeds.
   If R3 routed the decision to Arm B, use only Arm B for this decision.
   Propose more seeds.
6. **R6, no rescue.** If pure degrades with load (R3 is unmet), the transformer passes R2, and no eligible hybrid reaches the 10-point threshold in either seed, conclude that one attention layer does not rescue capacity at this scale.

Use unrounded accuracies at decision thresholds and require eligible runs in both seeds to classify a matched comparison as R4 or R5.
If failures or missing evidence make the next decision rule ambiguous, report what can be established and ask before interpreting further.
This is a two-seed descriptive pilot and has no significance tests.
Stop after reporting, with any future ICL inference variant requiring separate approval.
