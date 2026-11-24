# One Attention Layer vs. Pure Gated DeltaNet for In-Context Rule Learning

This repository holds the code and results for the study described in `RESEARCH_PROPOSAL.md`. The study asks whether replacing **one** Gated DeltaNet mixer with **one** causal softmax-attention layer helps a small recurrent language model infer the meaning of a made-up command from in-context demonstrations.

Where to find things:
- Final report: `reports/FINAL_REPORT.md`
- Plan: `IMPLEMENTATION_PLAN.md`, including the pre-registered analysis
- Chronological log: `EXPERIMENT_LOG.md`, including deviations from the plan

## Setup

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt   # exact versions: requirements.lock.txt
export OPENBLAS_NUM_THREADS=1   # recommended on a busy shared server
```

The Gated DeltaNet kernels come from Flash Linear Attention (`fla.layers.GatedDeltaNet`, v0.6.0) and need a CUDA GPU. The experiments ran on NVIDIA L40S GPUs.

## Reproducing everything

```bash
# 1. tests (CUDA tests are skipped if no GPU is visible)
.venv/bin/python -m pytest -q tests

# 2. fixed validation set + paired test cases (deterministic; hashes in data/data_manifest.json)
.venv/bin/python scripts/generate_data.py --config configs/base.yaml

# 3. (optional, saves CPU) pre-generate the deterministic training stream per seed
.venv/bin/python scripts/pregenerate_train.py --config configs/base.yaml --seeds 0 1 2 3 4

# 4. train + evaluate all 4 architectures x 5 seeds (GPU-cap-aware queue; resumable)
.venv/bin/python scripts/run_experiments.py --config configs/base.yaml --group main

# 5. statistics, tables, figures
.venv/bin/python scripts/analyze_results.py
.venv/bin/python scripts/generate_figures.py
```

Exploratory follow-up (post hoc): pure model, data-stream seed × init seed, validation only:

```bash
.venv/bin/python scripts/run_experiments.py --jobs-json configs/seedsplit_jobs.json --jobs-per-gpu 3
.venv/bin/python scripts/run_experiments.py --jobs-json configs/earlyinit_jobs.json --jobs-per-gpu 3   # experiment A
.venv/bin/python scripts/generate_figures.py      # adds fig9_data_vs_init_seed, fig10_early_init_transplant
```

`--early-init-seed S` transplants the GDN layer 0–1 decay and convolution initialisation from seed S (experiment A, report §10.1).

Experiment 2 (pre-registered in `IMPLEMENTATION_PLAN.md` §9): ambiguous demonstrations, each consistent with two operations:

```bash
.venv/bin/python scripts/generate_data.py --config configs/ambiguous.yaml          # -> data/amb/
.venv/bin/python scripts/run_experiments.py --config configs/ambiguous.yaml --group amb
.venv/bin/python scripts/analyze_exp2.py                                           # -> results/amb/
.venv/bin/python scripts/generate_figures.py --tables results/amb/tables --ckpt-dir checkpoints/amb --out results/amb/figures
```

`--data-seed` on `src.training.train` decouples the training-data stream from the initialisation seed. It defaults to `--seed`, as in all main runs.

**Headline result:** no hypothesis was supported in either pre-registered experiment.
- Experiment 1: rule inference was saturated at one demonstration, and a seed-level phase transition dominated.
- Experiment 2 (ambiguous demonstrations): inference was needed, but every model that learned the task, the pure GDN included, combined demonstrations near-optimally. Differences came only from which runs learned program execution within 20k steps.

See `reports/FINAL_REPORT.md`.

Single run / single evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.training.train --config configs/base.yaml \
    --arch hybrid_middle --seed 0 --out checkpoints/main/hybrid_middle_s0
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate_models.py --run checkpoints/main/hybrid_middle_s0
```

`--arch` is one of `pure`, `hybrid_early` (attention at layer 2 of 0–7), `hybrid_middle` (layer 4) or `hybrid_late` (layer 6). Training resumes automatically from `latest.pt`.

**Shared-server note:** `scripts/run_experiments.py` limits the user to **2 distinct GPUs in total**, counted across all of the user's processes and sessions. It stacks several small jobs on GPUs it already occupies, and never uses GPUs occupied by other people.

It also respects the user-wide CPU/memory budget (6-CPU quota, 175G memory throttle shared by all sessions):
- it runs at most `--max-jobs` jobs (default 2);
- it launches only when `~/.local/bin/eric-headroom` passes;
- it runs each job under `systemd-run --user --scope -p MemoryMax=4G -p TasksMax=128` with one CPU thread;
- it stops its newest job if headroom fails mid-run.

Measure one job before raising these limits (`/usr/bin/time -v`).

## Layout

```
configs/            base.yaml (all hyper-parameters), ambiguous.yaml (Experiment 2), pure_deltanet.yaml,
                    hybrid.yaml, calibration/, seedsplit_jobs.json + earlyinit_jobs.json (follow-ups)
src/data/           tokenizer.py, interpreter.py (independent reference), generator.py, dataset.py
src/models/         common.py (blocks, GDN wrapper, attention), pure_deltanet.py, hybrid.py
src/training/       train.py, checkpoint.py
src/evaluation/     evaluate.py, paired_accuracy.py, statistics.py
src/utils/          reproducibility.py, logging.py
scripts/            generate_data.py, pregenerate_train.py, run_experiments.py, evaluate_models.py,
                    analyze_results.py, analyze_exp2.py (Experiment 2), generate_figures.py
tests/              pytest suite (50 tests; CUDA tests skipped without a GPU)
data/               Experiment 1 val/test data; data/amb/ = Experiment 2; train_cache/ (regenerable)
checkpoints/        main/ (Exp 1), amb/ (Exp 2), seedsplit/ + earlyinit/ (exploratory follow-ups);
                    each run: {final.pt, latest.pt, metadata.json, train_log.jsonl}
results/raw/        Exp 1: one CSV per trained model, one row per evaluated prompt (raw_amb/ = Exp 2)
results/tables/     Exp 1 aggregated tables (csv + md); results/stats.json
results/figures/    Exp 1 + follow-up figures (PNG + PDF)
results/amb/        Exp 2 tables/, figures/, stats.json, stats_exp2.json
logs/               scheduler and per-run training logs
reports/            FINAL_REPORT.md
```

## What is (and isn't) in version control

`.gitignore` keeps the repository to about 90 MB.

**Tracked:**
- all code, configs, tests and documents;
- the fixed validation and test sets, with sha256 hashes in `data/**/data_manifest.json`;
- every raw prediction CSV (`results/raw*/`), from which all reported numbers are computed;
- tables, figures and `stats*.json`;
- for every run, `checkpoints/**/metadata.json` (config, parameter counts, environment) and `train_log.jsonl` (training curves).

**Not tracked (regenerable):**
- model weights `*.pt` (~6 GB): retrain with the commands above. Runs are deterministic in data and initialisation; kernels are not bit-exact (see report §13);
- `data/train_cache/` (~3.7 GB): `scripts/pregenerate_train.py`, or generated on the fly automatically;
- `logs/`: raw scheduler and training logs. The curated record is `EXPERIMENT_LOG.md`;
- `reports/notion_export*`: a derived export of the report;
- `.venv/`: rebuild from `requirements.lock.txt`.

**Shared-server safeguards:** `scripts/run_experiments.py` uses two lab-specific safeguards when they exist:
- `~/.local/bin/eric-headroom`, overridable with `HEADROOM_CMD`;
- `systemd-run --user` per-job caps.

On other machines both are skipped with a warning, and the 2-GPU cap logic still applies.
