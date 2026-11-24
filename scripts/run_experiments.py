"""Job queue for training + evaluation that enforces the lab's GLOBAL GPU cap.

    python scripts/run_experiments.py --config configs/base.yaml \
        --group main [--archs pure hybrid_early ...] [--seeds 0 1 2 3 4] [--steps N]

Every job (one training run followed by its test evaluation) is a separate
process.  Lab rule: the user may occupy at most 2 *distinct* GPUs at once, across
all projects and sessions (several processes on one GPU count once).  Before
each launch the scheduler determines the set of GPUs on which the user has any
process (nvidia-smi, all sessions) plus GPUs holding its own not-yet-initialised
jobs.  It launches on a new idle GPU only if that set has < 2 elements;
otherwise it stacks up to --jobs-per-gpu jobs onto GPUs the user already uses,
provided enough memory is free.  GPUs used by other people are never chosen.

Jobs are idempotent: finished runs (final.pt + raw csv) are skipped and
interrupted runs resume from their latest checkpoint.

CPU/memory rules (lab incident 2026-11-22, ~/.claude/CLAUDE.md): all of the user's processes share
one cgroup (6 CPUs, memory throttle at 175G, 4096 tasks).  Therefore:
  * at most --max-jobs of this scheduler's jobs run at once (default 2);
  * every launch requires `eric-headroom` to exit 0;
  * every job runs under `systemd-run --user --scope -p MemoryMax=.. -p TasksMax=..` (set from a
    measured single-job peak) with OMP/MKL/OPENBLAS threads = 1;
  * if eric-headroom fails while jobs run, the newest job of this scheduler is stopped (it resumes
    from its checkpoint later); a job is re-queued at most --max-attempts times, then dropped.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logging import load_config  # noqa: E402

MAX_USER_GPUS = 2          # lab rule: at most 2 *distinct* GPUs per user, all sessions combined
USER = getpass.getuser()
PY = sys.executable


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def gpu_table():
    """{index: dict(uuid, mem_used, mem_total, util, user_procs, other_procs)}"""
    gpus, by_uuid = {}, {}
    for line in sh("nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu "
                   "--format=csv,noheader,nounits").splitlines():
        i, uuid, mem, tot, util = [x.strip() for x in line.split(",")]
        gpus[int(i)] = {"uuid": uuid, "mem_used": int(mem), "mem_total": int(tot), "util": int(util),
                        "user_procs": 0, "other_procs": 0}
        by_uuid[uuid] = int(i)
    for line in sh("nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader").splitlines():
        try:
            pid, uuid = [x.strip() for x in line.split(",")]
        except ValueError:
            continue
        owner = sh(f"ps -o user= -p {pid}").strip()
        if uuid in by_uuid:
            gpus[by_uuid[uuid]]["user_procs" if owner == USER else "other_procs"] += 1
    return gpus


def choose_gpu(my_jobs_per_gpu: dict, max_gpus: int, jobs_per_gpu: int, min_free_mib: int,
               my_invisible_per_gpu: dict | None = None):
    """Return a GPU index to launch on without exceeding the user's distinct-GPU cap, or None.

    user GPUs = GPUs where the user has any process (any project/session) plus GPUs
    holding this scheduler's own jobs (which may not have initialised CUDA yet)."""
    g = gpu_table()
    user_gpus = {i for i, x in g.items() if x["user_procs"] > 0} | {i for i, n in my_jobs_per_gpu.items() if n > 0}
    free = lambda i: g[i]["mem_total"] - g[i]["mem_used"]
    # per-GPU load = all of the user's GPU processes visible there (any queue/session) plus this
    # scheduler's own jobs that have not initialised CUDA yet (not yet visible to nvidia-smi)
    inv = my_invisible_per_gpu or {}
    load = lambda i: g[i]["user_procs"] + inv.get(i, 0)
    # 1) stack onto a GPU the user already uses (does not raise the distinct-GPU count)
    stack = [i for i in user_gpus if load(i) < jobs_per_gpu and free(i) > min_free_mib
             and (g[i]["other_procs"] == 0 or i in my_jobs_per_gpu)]
    # 2) a new idle GPU, only if the cap allows
    new = []
    if len(user_gpus) < max_gpus:
        new = [i for i, x in g.items() if i not in user_gpus and x["other_procs"] == 0
               and x["mem_used"] < 2000 and x["util"] < 20]
    if new:  # spread first: prefer an idle new GPU while the cap allows it
        return sorted(new, key=lambda i: g[i]["mem_used"])[0], len(user_gpus) + 1
    if stack:
        return sorted(stack, key=lambda i: (load(i), -free(i)))[0], len(user_gpus)
    return None, len(user_gpus)


# Lab-specific budget check (all of the user's sessions share one cgroup). On machines without it
# the check is skipped with a warning; on the lab server it is mandatory and present.
HEADROOM = os.environ.get("HEADROOM_CMD", os.path.expanduser("~/.local/bin/eric-headroom"))
_warned = set()


def _warn_once(key: str, msg: str):
    if key not in _warned:
        _warned.add(key)
        print(f"[sched] WARNING: {msg}", flush=True)


def headroom_ok() -> bool:
    """True if the user-wide CPU/memory budget allows another launch."""
    if not os.path.exists(HEADROOM):
        _warn_once("headroom", f"{HEADROOM} not found; skipping the user-wide memory/CPU budget check")
        return True
    return subprocess.run([HEADROOM], capture_output=True).returncode == 0


def scope_prefix(mem_max: str, tasks_max: int) -> str:
    """Per-job hard memory/task cap via a systemd user scope, if available."""
    import shutil
    if shutil.which("systemd-run") is None or \
            subprocess.run(["systemd-run", "--user", "--scope", "-q", "true"], capture_output=True).returncode:
        _warn_once("scope", "systemd-run --user unavailable; jobs run without a per-job memory cap")
        return ""
    return f"systemd-run --user --scope -q -p MemoryMax={mem_max} -p TasksMax={tasks_max}"


def gpu_pids() -> set[int]:
    out = set()
    for line in sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader").split():
        if line.strip().isdigit():
            out.add(int(line))
    return out


def descendants(pid: int) -> set[int]:
    out = sh(f"pgrep -P {pid}").split()
    res = set()
    for c in out:
        res.add(int(c))
        res |= descendants(int(c))
    return res


def job_running_elsewhere(run: Path) -> bool:
    """Guard against duplicate launches (e.g. after a scheduler restart)."""
    return bool(sh(f"pgrep -u {USER} -f -- '--out {run}( |$)'").strip())


def build_jobs(a) -> list[dict]:
    """Job = dict(config, group, arch, seed, steps, eval).  Either from --jobs-json
    (list of such dicts) or from --config/--group/--archs/--seeds."""
    if a.jobs_json:
        specs = json.load(open(a.jobs_json))
    else:
        cfg = load_config(a.config)
        archs = a.archs or cfg["experiment"]["archs"]
        seeds = a.seeds if a.seeds is not None else cfg["experiment"]["seeds"]
        # seed-major order so that partial completion yields balanced architectures
        specs = [{"config": a.config, "group": a.group, "arch": arch, "seed": s, "steps": a.steps,
                  "eval": not a.no_eval} for s in seeds for arch in archs]
    jobs = []
    for j in specs:
        name = j.get("name") or (f"{j['arch']}_s{j['seed']}" +
                                 (f"_d{j['data_seed']}" if j.get("data_seed") is not None else ""))
        run = ROOT / "checkpoints" / j["group"] / name
        raw_dir = ROOT / "results" / ("raw" if j["group"] == "main" else f"raw_{j['group']}")
        done = (run / "final.pt").exists() and (not j.get("eval", True) or
                                                 (raw_dir / f"{name}.csv").exists())
        if not done:
            jobs.append({**j, "run": run, "raw_dir": raw_dir})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--group", default="main")
    ap.add_argument("--archs", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--jobs-json", help="JSON list of {config, group, arch, seed, steps, eval}")
    ap.add_argument("--max-gpus", type=int, default=MAX_USER_GPUS)
    ap.add_argument("--jobs-per-gpu", type=int, default=3)
    ap.add_argument("--min-free-mib", type=int, default=10000)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--no-eval", action="store_true", help="train only (pilots: never touch test data)")
    ap.add_argument("--max-jobs", type=int, default=2, help="max concurrent jobs of this scheduler")
    ap.add_argument("--mem-max", default="4G", help="systemd scope MemoryMax per job (~1.5x measured peak)")
    ap.add_argument("--tasks-max", type=int, default=128, help="systemd scope TasksMax per job")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--require-resource-guards", action="store_true",
                    help="require an executable headroom check and working resource scopes")
    ap.add_argument("--measure", action="store_true",
                    help="measure training resources with /usr/bin/time -v per job")
    a = ap.parse_args()
    assert a.max_gpus <= MAX_USER_GPUS
    jobs = build_jobs(a)
    print(f"[sched] {len(jobs)} jobs pending: {[(j['group'], j['arch'], j['seed']) for j in jobs]}", flush=True)
    if not jobs:
        return
    cap = scope_prefix(a.mem_max, a.tasks_max)
    if a.require_resource_guards and (not os.access(HEADROOM, os.X_OK) or not cap):
        ap.error("a working headroom command and systemd user scope are required")
    if a.measure and not os.access("/usr/bin/time", os.X_OK):
        ap.error("--measure requires /usr/bin/time")
    running: dict[int, tuple] = {}  # pid -> (popen, gpu, job)
    attempts: dict[str, int] = {}
    while jobs or running:
        if running and not headroom_ok():       # shed load: stop the newest job, re-queue it
            pid = max(running)
            p, gpu, tag = running.pop(pid)
            print(f"[sched] {time.strftime('%H:%M:%S')} eric-headroom FAILED: stopping newest job {tag}",
                  flush=True)
            try:
                os.killpg(pid, signal.SIGTERM)
                p.wait(timeout=60)
            except Exception:
                os.killpg(pid, signal.SIGKILL)
            job = p.job
            if attempts.get(str(job["run"]), 0) < a.max_attempts:
                jobs.insert(0, job)
            else:
                print(f"[sched] giving up on {tag} after {a.max_attempts} attempts", flush=True)
            time.sleep(120)
            continue
        for pid, (p, gpu, job) in list(running.items()):
            if p.poll() is not None:
                print(f"[sched] {time.strftime('%H:%M:%S')} finished {job} on GPU {gpu} rc={p.returncode}",
                      flush=True)
                del running[pid]
        if jobs and len(running) < a.max_jobs and headroom_ok():
            per_gpu = {}
            for _, g_, _ in running.values():
                per_gpu[g_] = per_gpu.get(g_, 0) + 1
            vis = gpu_pids()
            invisible = {}
            for pid_, (_, g_, _) in running.items():
                if not (({pid_} | descendants(pid_)) & vis):
                    invisible[g_] = invisible.get(g_, 0) + 1
            gpu, n_user_gpus = choose_gpu(per_gpu, a.max_gpus, a.jobs_per_gpu, a.min_free_mib,
                                          invisible)
            if gpu is not None:
                j = jobs.pop(0)
                run, tag = j["run"], (j["group"], j["arch"], j["seed"])
                if job_running_elsewhere(run):
                    print(f"[sched] skip {tag}: already running in another process", flush=True)
                    continue
                steps = f"--steps {j['steps']}" if j.get("steps") else ""
                if j.get("data_seed") is not None:
                    steps += f" --data-seed {j['data_seed']}"
                steps += " " + j.get("extra", "")
                log_dir = ROOT / "logs" / j["group"]
                log_dir.mkdir(parents=True, exist_ok=True)
                measure = f"/usr/bin/time -v -o {log_dir / (run.name + '_time.txt')} " if a.measure else ""
                cmd = (f"{cap} {measure}{PY} -m src.training.train --config {j['config']} --arch {j['arch']} "
                       f"--seed {j['seed']} {steps} --out {run}")
                if j.get("eval", True):
                    task = load_config(j["config"])["data"].get("task", "program")
                    ev = {"recall": "evaluate_recall.py", "mqar": "evaluate_mqar.py"}.get(task, "evaluate_models.py")
                    cmd += f" && {cap} {PY} scripts/{ev} --run {run} --raw-dir {j['raw_dir']}"
                attempts[str(run)] = attempts.get(str(run), 0) + 1
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           TORCHINDUCTOR_COMPILE_THREADS="1")
                logf = open(log_dir / f"{run.name}.log", "a")
                p = subprocess.Popen(cmd, shell=True, cwd=ROOT, env=env, stdout=logf,
                                     stderr=subprocess.STDOUT, start_new_session=True)
                p.job = j
                running[p.pid] = (p, gpu, tag)
                print(f"[sched] {time.strftime('%H:%M:%S')} launched {tag} on GPU {gpu} "
                      f"(distinct user GPUs after launch: {n_user_gpus}; my jobs on GPU: "
                      f"{per_gpu.get(gpu, 0) + 1})", flush=True)
                time.sleep(30)  # let it initialise CUDA so it is visible to nvidia-smi
                continue
        time.sleep(a.poll)
    print("[sched] all jobs done", flush=True)


if __name__ == "__main__":
    main()
