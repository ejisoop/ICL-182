"""Training entry point.

    python -m src.training.train --config configs/base.yaml --arch pure --seed 0 \
        --out checkpoints/pure_s0

Resumes automatically from ``<out>/latest.pt`` if present.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import TokenListDataset, TrainStream, collate_lm, load_val
from src.data.generator import TrainSampleConfig
from src.data.tokenizer import VOCAB_SIZE, encode
from src.models import ModelConfig, build_model
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logging import JsonlLogger, load_config
from src.utils.reproducibility import environment_info, set_seed


EARLY_KEYS = re.compile(r"blocks\.[01]\.mixer\.inner\.(A_log|dt_bias|[qkv]_conv1d\.weight)$")


def is_recall(cfg: dict) -> bool:
    return cfg["data"].get("task", "program") == "recall"


def model_config_from(cfg: dict) -> ModelConfig:
    if is_recall(cfg):
        from src.data import recall
        return ModelConfig(vocab_size=recall.VOCAB_SIZE, **cfg["model"])
    return ModelConfig(vocab_size=VOCAB_SIZE, **cfg["model"])


def lr_at(step: int, tc: dict) -> float:
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    prog = (step - tc["warmup_steps"]) / max(1, tc["steps"] - tc["warmup_steps"])
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return tc["lr"] * (tc["min_lr_ratio"] + (1 - tc["min_lr_ratio"]) * cos)


def make_optimizer(model, tc):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 and "embed" not in n else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=tc["lr"], betas=tuple(tc["betas"]), fused=torch.cuda.is_available())


@torch.no_grad()
def evaluate_val(model, val_rows, device, amp_dtype, batch_size=256) -> dict:
    model.eval()
    order = sorted(range(len(val_rows)), key=lambda i: len(val_rows[i]["ids"]))
    tot = {"all": [0.0, 0, 0], "icl": [0.0, 0, 0], "definition": [0.0, 0, 0]}
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        b = collate_lm([val_rows[i]["ids"] for i in idx])
        inp = b["input_ids"].to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(inp)
        ar = torch.arange(len(idx), device=device)
        la = logits[ar, b["answer_pos"].to(device)].float()
        ans = b["answer"].to(device)
        ce = F.cross_entropy(la, ans, reduction="none")
        corr = (la.argmax(-1) == ans)
        for j, i in enumerate(idx):
            for key in ("all", val_rows[i]["format"]):
                tot[key][0] += ce[j].item()
                tot[key][1] += int(corr[j].item())
                tot[key][2] += 1
    model.train()
    out = {}
    for k, (l, c, n) in tot.items():
        if n:
            out[f"val_loss_{k}"] = l / n
            out[f"val_acc_{k}"] = c / n
    return out


def train(config_path: str, arch: str, seed: int, out_dir: str, overrides: dict | None = None,
          device: str = "cuda", data_seed: int | None = None,
          early_init_seed: int | None = None) -> dict:
    """``seed`` sets initialisation (and RNG); the training-data stream uses ``data_seed``,
    which defaults to ``seed`` (as in all main runs)."""
    cfg = load_config(config_path, overrides)
    data_seed = seed if data_seed is None else data_seed
    cfg["arch"], cfg["seed"], cfg["data_seed"] = arch, seed, data_seed
    tc, dc = cfg["train"], cfg["data"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    amp_dtype = getattr(torch, tc["amp_dtype"])

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))  # lab quota: 6 CPUs total
    set_seed(seed)
    mcfg = model_config_from(cfg)
    model = build_model(arch, mcfg)
    if early_init_seed is not None:
        # exploratory: transplant the initial GDN decay/conv parameters of layers 0-1 from a
        # model built with another seed (everything else keeps the `seed` initialisation)
        set_seed(early_init_seed)
        donor = build_model(arch, mcfg).state_dict()
        sd = model.state_dict()
        keys = [k for k in sd if EARLY_KEYS.match(k)]
        assert keys, "no early GDN parameters found"
        model.load_state_dict({**sd, **{k: donor[k] for k in keys}})
        cfg["early_init_seed"], cfg["early_init_keys"] = early_init_seed, keys
        set_seed(seed)
    model = model.to(dev)
    opt = make_optimizer(model, tc)
    n_params = model.num_params()

    start_step, elapsed_prev = 0, 0.0
    if (out / "latest.pt").exists():
        st = load_checkpoint(out / "latest.pt", model, opt, map_location=dev)
        start_step = st["step"]
        elapsed_prev = st["extra"].get("elapsed", 0.0)
        print(f"[resume] {out} from step {start_step}", flush=True)
    if start_step >= tc["steps"] and (out / "final.pt").exists():
        print(f"[skip] {out} already complete", flush=True)
        return json.load(open(out / "metadata.json"))

    meta = {"arch": arch, "seed": seed, "data_seed": data_seed, "n_params": n_params,
            "param_breakdown": model.param_breakdown(),
            "attention_layers": model.attention_layer_indices(),
            "layer_types": mcfg.layer_types(), "config": cfg, "env": environment_info()}
    json.dump(meta, open(out / "metadata.json", "w"), indent=2)

    val_rows = load_val(dc["val_file"])
    remaining = tc["steps"] - start_step
    if is_recall(cfg):   # load-scaling task (IMPLEMENTATION_PLAN section 11)
        from src.data import recall
        for r in val_rows:
            r["ids"] = recall.encode(r["text"])
        rcfg = recall.RecallConfig(n_choices=tuple(dc["n_choices"]), n_query=dc["n_query"],
                                   p_definition=dc["p_definition"])
        stream = recall.RecallStream(data_seed, rcfg, start_index=start_step * tc["batch_size"],
                                     length=remaining * tc["batch_size"])
    else:
        for r in val_rows:
            r["ids"] = encode(r["text"])
        scfg = TrainSampleConfig(k_min=dc["k_min"], k_max=dc["k_max"], p_definition=dc["p_definition"],
                                 n_init=dc["n_init"], n_stmts=dc["n_stmts"],
                                 ambiguous_demos=dc.get("ambiguous_demos", False))
        stream = TrainStream(data_seed, scfg, start_index=start_step * tc["batch_size"],
                             length=remaining * tc["batch_size"], cache_dir=dc.get("cache_dir"))
    print(f"[data] training stream cached={stream.cached}", flush=True)
    nw = int(dc["num_workers"])
    loader = DataLoader(stream, batch_size=tc["batch_size"], shuffle=False,
                        num_workers=nw, collate_fn=collate_lm, pin_memory=True,
                        persistent_workers=False, prefetch_factor=(8 if nw > 0 else None))
    log = JsonlLogger(out / "train_log.jsonl")
    model.train()
    t0 = time.time()
    step = start_step
    run_loss = run_ans = 0.0
    n_run = 0
    for batch in loader:
        lr = lr_at(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        inp = batch["input_ids"].to(dev, non_blocking=True)
        labels = batch["labels"].to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=dev.type == "cuda"):
            logits = model(inp)
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        with torch.no_grad():
            ar = torch.arange(inp.shape[0], device=dev)
            ans_loss = F.cross_entropy(logits[ar, batch["answer_pos"].to(dev)],
                                       batch["answer"].to(dev))
        run_loss += loss.item(); run_ans += ans_loss.item(); n_run += 1
        step += 1
        if step % tc["log_interval"] == 0 or step == tc["steps"]:
            el = elapsed_prev + time.time() - t0
            log.log({"step": step, "train_loss": run_loss / n_run, "train_answer_loss": run_ans / n_run,
                     "lr": lr, "grad_norm": gnorm.item(), "elapsed_s": el})
            run_loss = run_ans = 0.0; n_run = 0
        if step % tc["eval_interval"] == 0 or step == tc["steps"]:
            v = evaluate_val(model, val_rows, dev, amp_dtype)
            v.update({"step": step, "elapsed_s": elapsed_prev + time.time() - t0})
            log.log(v)
            print(f"[{arch} s{seed}] step {step} " + " ".join(
                f"{k}={x:.4f}" for k, x in v.items() if k.startswith("val_")), flush=True)
        if step % tc["ckpt_interval"] == 0 or step == tc["steps"]:
            save_checkpoint(out / "latest.pt", model, opt, step,
                            {"elapsed": elapsed_prev + time.time() - t0})
    total = elapsed_prev + time.time() - t0
    save_checkpoint(out / "final.pt", model, None, step, {"elapsed": total})
    meta.update({"final_step": step, "train_seconds": total})
    json.dump(meta, open(out / "metadata.json", "w"), indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override train.steps")
    ap.add_argument("--data-seed", type=int, default=None,
                    help="training-data stream seed (default: = --seed)")
    ap.add_argument("--early-init-seed", type=int, default=None,
                    help="exploratory: take layer-0/1 GDN A_log, dt_bias, conv init from this seed")
    a = ap.parse_args()
    ov = {"train": {"steps": a.steps}} if a.steps else None
    train(a.config, a.arch, a.seed, a.out, ov, data_seed=a.data_seed,
          early_init_seed=a.early_init_seed)


if __name__ == "__main__":
    main()
