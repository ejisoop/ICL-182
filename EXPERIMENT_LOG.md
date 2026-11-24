# Experiment Log

Chronological log of what was actually done. All times are server local (2026-11-18 onward).

## 2026-11-18 — Phase 1: environment
- The repository contained only `CLAUDE_CODE_PROMPT.md` and `RESEARCH_PROPOSAL.md`.
- Hardware: 7× L40S, shared with other users. GPUs 0–3 were busy with other users' jobs; GPUs 4–6 were idle at inspection.
- No torch was installed in the system Python. I created a uv venv `.venv` (Python 3.11) with torch 2.7.1+cu126, triton 3.3.1 and flash-linear-attention 0.6.0. The full lock file is in `requirements.lock.txt`.
- `fla.layers.GatedDeltaNet` forward and backward ran on an L40S in bf16. **No substitution of the Gated DeltaNet implementation was needed.**

## Phase 2–3: task, data, models
- Implemented the tokenizer, an independent text-parsing interpreter, the generator with template-hash splits, and the models: pure GDN and hybrids with attention at layer 2, 4 or 6.
- Parameter counts: pure 7,015,232; hybrids 6,944,504 (−1.0%).
- Generated `data/val.jsonl` (3000 prompts) and `data/test_cases.jsonl` (1000 paired cases). SHA-256 hashes are in `data/data_manifest.json`.
- Test suite: 38 tests pass. There were 2 initial failures, both bugs in the tests themselves:
  - One hand-written demonstration was ambiguous: `b=1, dax b 4` gives 4 under both `set` and `mul`.
  - One assertion was malformed.
  - Neither was a bug in the implementation.

## Phase 4: pilot
- At the time of launch, eric already had 2 GPU processes from a different project and session (ReCoLab), so the global cap was full.
- The pilot scheduler (`scripts/run_experiments.py --group pilot`) waits until the cap allows a launch.

### Pilot v1 (3 inits + 4 updates; seed 100; pure and hybrid_middle) — stopped at ~5k of 10k steps
- Throughput was about 0.12–0.16 s/step on an L40S (batch 128).
- Validation answer accuracy after 4k steps:
  - pure: ICL 20.5%, definition 22.2%
  - hybrid_middle: ICL 21.7%, definition 22.7%
  - Chance is 7.7% (1/13).
- Train loss on demonstration-result tokens fell to about 1.0–1.3, while answer loss stayed around 2.1–2.35.
- **Diagnosis:** the models learn single-operation demonstrations, but executing a 4-update program with variable cross-references is the bottleneck. This holds even in the definition condition, where no inference is needed. Both architectures would sit near a floor, and the comparison would be uninformative.
- **Protocol change, decided before any main run and recorded here:**
  - Query complexity becomes a config parameter (`data.n_init`, `data.n_stmts`).
  - The template now also encodes data flow: which variable slot each statement updates and reads. This keeps template-level held-out splits meaningful for short programs.
- **Calibration rule, stated before the calibration runs:**
  - Train **pure only**, seed 100, 6k steps, for 2 inits + 2 updates and for 2 inits + 3 updates.
  - Choose the *hardest* setting whose pure-model **definition-control** validation accuracy at 6k steps is ≥ 60%.
  - The calibration is blind to the hypothesis: it uses no hybrid and no ICL comparison. Pilots never run test evaluation (`--no-eval`).
- v1 pilot outputs are archived in `checkpoints/pilot_v1/` and `logs/pilot_v1/`.

### Calibration round 1 (pure, seed 100, 6k steps; validation set only)
| task (inits×updates) | val acc ICL | val acc definition |
|---|---|---|
| 2×2 | 38.9% | 47.1% |
| 2×3 | 36.6% | 36.1% |

- Neither setting reached the ≥60% threshold. Both were still improving when the cosine schedule ended.
- Diagnostic on the 2×2 model, validation set only: definition accuracy by operation is add 36%, mul 48%, set 48%, sub 56%. The model is undertrained on mod-13 arithmetic composition. There is no evidence of a pipeline bug.
- Infrastructure changes:
  - The lab GPU rule changed to *at most 2 distinct GPUs* (several processes on one GPU count once). The scheduler was rewritten to enforce this and to stack jobs per GPU.
  - Each seed's training stream is now cached to disk once and shared by the 4 architectures. The content is identical to on-the-fly generation (verified by test).
  - OpenBLAS and Inductor thread caps were added after thread-creation failures under heavy server load (load average 100–240).
- The Claude CLI crashed once mid-session. All processes and files survived, and work resumed.

### Calibration round 2 (pure, seed 100, **20k steps** = candidate main budget; validation set only)
- Grid: {2×2, 2×3} × lr {1e-3, 2e-3}.
- **Rule (stated before launch):** choose the hardest task whose pure definition-control validation accuracy at 20k steps is ≥ 70%, at its better learning rate. If none qualifies, choose the configuration with the highest definition accuracy.
- The hybrid is not involved, and test data is never touched.

**Round 2 results** (final step 20k; validation set: ~2.5k ICL / ~0.5k definition prompts)
| task | lr | val acc ICL | val acc definition |
|---|---|---|---|
| 2×2 | 1e-3 | 49.7% | 51.0% |
| 2×2 | 2e-3 | 50.3% | 53.7% |
| 2×3 | 1e-3 | 48.0% | **55.4%** |
| 2×3 | 2e-3 | 48.4% | 54.8% |

- No configuration reached 70%, so the fallback clause applies: **2×3 with lr 1e-3** has the highest definition accuracy.
- The four configurations are within about 4 points of each other. With roughly 500 definition validation prompts, that gap is within sampling noise (SE ≈ 2.2 pp), so the choice is effectively a tie-break. It is applied as pre-stated and not re-litigated.
- All curves were still rising at 20k steps, so models are **not converged** at the main budget. The main budget is kept at 20k steps (the budget the rule was stated for) because of compute and time limits on the shared server. This is recorded as a limitation: absolute accuracies are moderate (about 50% per prompt, so paired accuracy will be lower), and the architecture comparison is made at a fixed, non-asymptotic budget.
- **Final main config:** `n_init=2`, `n_stmts=3`, `lr=1e-3`, `steps=20000`, all else as in `configs/base.yaml`.
- Fixed val/test data are regenerated with this config: `data/val.jsonl` and `data/test_cases.jsonl` (1000 cases). The old v1 files are overwritten; v1 was never used for any reported result.

## Phase 5: main experiment
- 21:5x: fixed data regenerated (`data/data_manifest.json`; test sha256 `7f61fd65…`). Training cache pre-generated for seeds 0–4 (`data/train_cache/498feb34db_s*`).
- 22:01: scheduler launched with `scripts/run_experiments.py --config configs/base.yaml --group main --jobs-per-gpu 4`. eric was using 0 GPUs at launch; jobs are placed on GPUs 4 and 5, for at most 2 distinct GPUs.
  - Throughput is about 0.6 s/step with 4 jobs per GPU.
  - The seed-1 jobs started before the seed-1 cache was written, so they generate data on the fly. The content is identical (tested); it only costs more CPU.
- Before any main result existed, the analysis and figure scripts were dry-run on *synthetic* raw files, written to the scratchpad and not to `results/`. Two cosmetic figure fixes followed (legend placement, log-axis tick labels). No analysis code was changed.
- 04:42 (11-19): all 20 main runs finished (rc=0) and were evaluated on the fixed test set. Results are in `results/raw/`.
- 04:50: the pre-registered analysis was run (`scripts/analyze_results.py`). **No hybrid-vs-pure difference** on any endpoint: all Holm p = 1 and all seed-paired p > 0.14. Placement ANOVA p = 0.96. No H3 interaction.
- **Unexpected observations, reported as observed:**
  1. **Bimodal outcome determined by seed, not architecture.** Seeds 1 and 4 "take off" (paired accuracy 68–90%) in **all four** architectures. Seeds 0, 2 and 3 plateau at about 21–26% in all four. This drives the very wide seed CIs.
  2. **ICL equals the definition control from k = 1 onward.** Paired accuracy is flat in k = 1…8 and equal to definition paired accuracy, within about 1 pp, in every run. A single individually unambiguous demonstration suffices to identify the operation, so the task as calibrated does not stress rule inference. The bottleneck is program execution.

### Exploratory follow-up (post hoc; NOT pre-registered): data stream vs initialisation
- **Question:** is the seed effect in (1) caused by the training-data stream or by the initialisation? Seed fixes both in the main design.
- **Design:** pure only, 20k steps, validation only (`--no-eval`-style, test set untouched). Data-stream seed ∈ {0 (failed in all main runs), 1 (succeeded in all)} × init seed ∈ {10, 11, 12}, giving 6 runs. Jobs are listed in `configs/seedsplit_jobs.json`.
- **Reading, stated before the runs:**
  - If take-off (final validation definition accuracy > 70%) tracks the data seed, the main comparison is effectively governed by a data-order lottery.
  - If it varies across init seeds within a data stream, initialisation matters.
- **Code change:** `train.py` gained `--data-seed` (default = `--seed`, so main runs are unaffected). `run_experiments.py` names such runs `<arch>_s<seed>_d<data_seed>`.
- 06:39: follow-up finished (6/6, rc=0). Final validation definition accuracy:

  | init | data stream 0 | data stream 1 |
  |---|---|---|
  | 10 | 50% | 49% |
  | 11 | 50% | 87% |
  | 12 | 83% | 53% |

  **Take-off does not track the data stream.**
- Check: all 4 architectures were built with the same seed and compared. 99.8% of initial parameters differ. Shared within a seed are only the data, the norm constants, and GDN layers 0–1 `A_log`, `dt_bias` and short-conv weights. The candidate cause (untested) is the initialisation of the early-layer decay and convolution parameters.
- Added figures 8 (per-seed curves) and 9 (follow-up). Report §10 filled in.

## Phase 6: final verification
- Test suite after all code changes: 33 passed on CPU (10 CUDA-skipped). The 11 CUDA model tests passed on GPU 4. Total: 44/44.
- `scripts/run_experiments.py --group main` reports 0 pending jobs (idempotent).
- Re-running `scripts/analyze_results.py` from `results/raw/` reproduces `results/stats.json` and `h1_comparisons.csv` byte-for-byte (md5 check).
- `scripts/generate_figures.py` regenerates all 9 figures (PNG + PDF), and each was inspected visually.

## Phase 7: continued work (user asked to keep going), 11-19
### Exploratory experiment A: does early GDN initialisation decide take-off? (post hoc; design fixed before runs)
- **Setup:** pure model, data stream 1 throughout (same as the §10 follow-up).
- **Code:** `--early-init-seed S` transplants `blocks.{0,1}.mixer.inner.{A_log, dt_bias, q/k/v_conv1d.weight}` from seed S's initialisation. Everything else is the `--seed` initialisation. Verified by `test_early_init_transplant`.
- **Known from the follow-up (data 1):** init 11 takes off; inits 10 and 12 plateau.
- **Runs** (`configs/earlyinit_jobs.json`):
  - body 10 + early 11
  - body 12 + early 11
  - body 11 + early 10
  - body 11 + early 12
  - reruns of (11, 11) and (10, 10), to measure run-to-run non-determinism of the fla kernels
- **Prediction if early-layer init is decisive:** both early-11 transplants take off (final validation definition accuracy > 70%), and both body-11 runs with early 10 or 12 plateau.
  - If instead take-off follows the body, the hypothesis is rejected.
  - If the reruns disagree with their originals, take-off is partly stochastic beyond the seed.
- **Result (15:23; final validation accuracy on definition prompts, ICL in brackets):**

  | body seed | early seed | definition | ICL | take-off? |
  |---|---|---|---|---|
  | 10 | 11 | 89.4 | 82.0 | yes |
  | 12 | 11 | 47.9 | 46.3 | no |
  | 11 | 10 | 80.1 | 72.2 | yes |
  | 11 | 12 | 96.8 | 89.9 | yes |
  | 11 | 11 (rerun) | 87.1 | 79.3 | yes (original 87.1 / 79.3, identical to 3 digits) |
  | 10 | 10 (rerun) | 50.1 | 45.4 | no (original 48.8 / 45.6) |

- **Verdict: the prediction is not met; the early-layer hypothesis is rejected as a sufficient explanation.** Both body-11 runs took off despite foreign early tensors (prediction: plateau). Only one of the two early-11 transplants took off.
  - Take-off is not decided by the early tensors alone, and not purely by the body either: replacing the 6 small early tensors flipped body 10 from plateau to take-off.
  - Reruns reproduce their originals closely, so outcomes are essentially deterministic given the init and data; the fla kernels' non-determinism is too small to flip them.
  - Reading: take-off is a sensitive function of the whole initialisation (small perturbations can flip it, but not always). Why the 4 architectures agreed within each main seed remains unexplained.

### Experiment 2 (ambiguous demonstrations): launch
- Pre-registered in `IMPLEMENTATION_PLAN.md` §9 before any run.
- Fixed data in `data/amb/` (test sha256 `60253b96…`). Training cache in `data/train_cache/2a07df57a2_s*`.
- New generator tests: each ambiguous demonstration is consistent with exactly 2 operations, with confusers uniform. Pairs stay valid. Identifiability by k matches 1 − 3^(1−k). The original task is unchanged byte-for-byte (manifest hashes reproduced).
- 12:40: the scheduler was launched alongside experiment A.
  - **Scheduler bug found and fixed:** the per-GPU stacking limit used `max(own jobs, user processes)`, which undercounted. One job (`amb/hybrid_early_s0`, <2 min old) landed as the 5th process on GPU 4.
  - That job and the scheduler were stopped. The scheduler now counts the user's visible GPU processes plus its own jobs that have not yet initialised CUDA.
  - The distinct-GPU cap (2) was never exceeded.
- 12:45: the scheduler was restarted. It skips the two already-running seed-0 jobs and resumes the rest.

### Experiment 2: freeze, resume bug, relaunch (11-19 / 11-20)
- **Freeze:** at 2026-11-19 16:38:50–16:39:09 all 8 running experiment-2 jobs and their scheduler stopped writing at the same moment (steps ~7.9k–9.4k).
  - No traceback, no out-of-memory error, disk fine. Most likely the whole process group was suspended along with the CLI session that launched it (not verifiable after the fact).
  - The processes held GPUs 4/5 without progress until they were terminated at 2026-11-20 11:15 (~18.5 h lost).
  - The completion waiter only counted result files, so it did not detect the stall.
- **Resume bug:** resuming from `latest.pt` crashed (`TypeError: RNG state must be a torch.ByteTensor`). `load_checkpoint` used `map_location="cuda"`, which moved the saved RNG states onto the GPU. Fixed by restoring them as CPU tensors; regression test `tests/test_checkpoint.py`.
  - The main experiment never resumed a run, so it is unaffected.
- **Relaunch (11:20):** the scheduler is now started with `setsid` (detached from the session), with jobs-per-gpu 5. Interrupted runs resume from step 8000, and the waiter now also stops on 30 min without file activity.
  - Resumed runs have duplicate train-log entries for steps 8000–~9350; readers keep the latest entry per step.
  - Training data and model state resume exactly from the step-8000 checkpoint, so outcomes are unaffected apart from kernel non-determinism.
- 2026-11-20 18:18: all 20 experiment-2 runs finished (rc=0) and were evaluated.

### Experiment 2: results (analysis exactly as pre-registered in §9; `scripts/analyze_exp2.py`)
- Primary (seed-paired t, Holm over 3): early −9.4 pp (p = 1.0), middle +8.0 (p = 1.0), late +9.6 (p = 1.0). Small-k also null. H2: Friedman p = 0.25. H3: interactions −1.4…+2.5 pp, Holm p = 1.0. **H1–H3 not supported.**
- Take-offs (definition individual accuracy > 70%): pure s4; middle s0, s1; late s1, s4; early none. All-4 agreement in 2/5 seeds, matching independence at the 25% rate (expected 1.6). The Experiment-1 within-seed sharing did not replicate.
- Inference is exercised: accuracy rises from k = 1 to k ≈ 4. ICL is 6–10 pp below definition on the primary endpoint.
- Exploratory (post hoc; added to `analyze_exp2.py` → `exp2_runs.md`): take-off runs are near-ideal Bayesian combiners. Identifiable-prompt accuracy 86.1% vs definition 86.8%. At k = 1, accuracy is 1.01–1.19 × half the definition accuracy. The pure GDN run (s4) is as good as the hybrids.
- Fig 8's title said "success is shared across architectures within a seed", which is false for Experiment 2; changed to a neutral title for both experiments.

## Phase 8: load-scaling pilot (user approved 2026-11-20)
- **Motivation** (report §12): in Experiments 1–2, program execution (the take-off lottery), not inference, decided outcomes, and 3 commands never stressed a recurrent state. The user asked whether the direction is "cooked". I proposed one pilot that stresses binding capacity with trivial execution, and the user approved it.
- **Built before any pilot run:**
  - `src/data/recall.py`: new task and its own vocabulary, with 512 made-up command tokens and an independent solver;
  - `configs/recall.yaml`;
  - `scripts/generate_recall_data.py`: fixed `data/recall/` sets; val sha256 `14d89286…`, test `fe75dce5…`;
  - `scripts/evaluate_recall.py`: scores every query;
  - `scripts/analyze_recall.py`: dry-run on synthetic data in the scratchpad recovered the injected effects;
  - tests `tests/test_recall.py` (9 pass);
  - `data.task: recall` switch in `train.py`; the scheduler picks the evaluator by task.
- Smoke test (speed only, seed 99, scratchpad, not analysed): the first attempt was killed by a CLI crash after 100 steps (both processes SIGKILLed together; not OOM). It was relaunched detached.
  - The first 100 steps took ~2.7 s/step, which includes Triton autotuning per new sequence length. The steady-state speed decides the step budget (below).
- Steady state measured: ~0.49 s/step (pure) and 0.53 s/step (hybrid_middle) with 2 jobs on one GPU. Budget kept at 20k steps.
- 00:05: pilot pre-registered in `IMPLEMENTATION_PLAN.md` §11, including the decision rules for "cooked" vs alive. Launched 4 archs × seeds 0, 1 with the GPU-cap scheduler (detached with `setsid`).
- Note: a GPU idle reaper (`~/.local/bin/gpu-idle-reaper`) started at 2026-11-21 00:00, after all earlier incidents. It kills eric's GPU processes that sit at 0% for 15 min. Jobs here compute continuously and checkpoint every 2000 steps.

### Phase 8 incident: user-slice resource exhaustion (2026-11-21 → 11-23)
- **What happened:** all of eric's processes (every session, tmux, cron) share one cgroup, `user-1013.slice`: 6-CPU quota, memory throttle at 175G (hard max 200G), 4096 tasks.
  - On 11-20, 11-21 and 11-22, several sessions ran detached jobs at the same time: this project's 8 training runs with 4 data-loader workers each, and another project's 32-worker data collection.
  - Together they pushed the slice into its memory throttle. Every process stalled: Claude CLI, GPU jobs, `nvidia-smi`, and new logins. On 11-22 the user was locked out.
  - In hindsight this also explains the Experiment-2 freeze (11-19 16:39) and the "Terminated" pilot launch (11-21 00:11). My earlier "suspended with the session" explanation was wrong.
- **My share:** 8 concurrent jobs × (main process + 4 loader workers that copy the parent) with OMP 4. That was sized from `nproc`=128, not from the 6-CPU quota, and the jobs were never measured before fan-out.
- **Fixes (following the new rules in ~/.claude/CLAUDE.md):**
  - recall config `num_workers: 0`: data generation is 6 ms/batch, so each job is one process. `train.py` uses `OMP_NUM_THREADS` (default 1) threads; the DataLoader supports 0 workers.
  - Sized one job before fan-out (`/usr/bin/time -v`, hybrid_middle, 400 steps, alone on GPU 2): peak RSS 1.66 GB, 5 threads, ~0.8 CPU, ~0.17 s/step (so ~1 h per 20k-step run).
  - `scripts/run_experiments.py`:
    - `--max-jobs` (default 2) concurrent jobs;
    - `eric-headroom` must pass before every launch;
    - every job runs under `systemd-run --user --scope -p MemoryMax=4G -p TasksMax=128`, with all BLAS/OMP threads = 1;
    - load shedding: if headroom fails while jobs run, stop the newest job (it resumes from its checkpoint) and re-queue it at most 3 times.
  - Pilot concurrency: 2 jobs (one per GPU), ~1.6 CPUs, because the other project was using ~3.7 of the 6 CPUs at launch time.
- The aborted 11-21 pilot directories (no checkpoints, no evaluations) were moved to `checkpoints/_aborted/recall_killed_20261121`.
