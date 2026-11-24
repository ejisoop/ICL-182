"""Publication figures (PNG + PDF) from results/tables and training logs.

    python scripts/generate_figures.py [--ckpt-dir checkpoints/main]

Run scripts/analyze_results.py first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.logging import read_jsonl  # noqa: E402

ARCH_ORDER = ["pure", "hybrid_early", "hybrid_middle", "hybrid_late"]
LABEL = {"pure": "Pure GDN", "hybrid_early": "Hybrid, attn @ L2 (early)",
         "hybrid_middle": "Hybrid, attn @ L4 (middle)", "hybrid_late": "Hybrid, attn @ L6 (late)"}
SHORT = {"pure": "Pure", "hybrid_early": "Early", "hybrid_middle": "Middle", "hybrid_late": "Late"}
# reference categorical palette (fixed order, slots 1-4) + secondary encoding by marker
COLOR = {"pure": "#2a78d6", "hybrid_early": "#eb6834", "hybrid_middle": "#1baf7a", "hybrid_late": "#eda100"}
MARKER = {"pure": "o", "hybrid_early": "s", "hybrid_middle": "D", "hybrid_late": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 6,
    "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print("wrote", out / name)


def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return sps.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def acc_vs_k(per_run, metric, title, ylabel, out, name):
    icl = per_run[per_run.condition == "icl"]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for arch in [a for a in ARCH_ORDER if a in set(icl.arch)]:
        g = icl[icl.arch == arch].groupby("k")[metric]
        ks = np.array(sorted(icl.k.unique()))
        m = g.mean().reindex(ks).values * 100
        h = np.array([tci(g.get_group(k).values) for k in ks]) * 100
        ax.fill_between(ks, m - h, m + h, color=COLOR[arch], alpha=0.15, linewidth=0)
        ax.plot(ks, m, color=COLOR[arch], marker=MARKER[arch], label=LABEL[arch],
                markeredgecolor="white", markeredgewidth=1)
    ax.set_xlabel("Demonstrations per command (k)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(icl.k.unique()))
    ax.set_ylim(-2, 102)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.0, -0.2, "Lines: mean over 5 training seeds; bands: 95% t-CI across seeds. 1000 paired test cases.",
            transform=ax.transAxes, fontsize=7.5, color=INK2)
    save(fig, out, name)


def diff_vs_k(per_run, out):
    icl = per_run[per_run.condition == "icl"]
    if "pure" not in set(icl.arch):
        return
    ks = np.array(sorted(icl.k.unique()))
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.axhline(0, color=INK2, linewidth=1)
    hybrids = [a for a in ARCH_ORDER[1:] if a in set(icl.arch)]
    offs = np.linspace(-0.15, 0.15, len(hybrids))
    for off, arch in zip(offs, hybrids):
        d, lo, hi = [], [], []
        for k in ks:
            a = icl[(icl.arch == arch) & (icl.k == k)]["paired_acc"].values
            b = icl[(icl.arch == "pure") & (icl.k == k)]["paired_acc"].values
            diff = a.mean() - b.mean()
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            if se > 0:
                dof = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
                    (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
                h = sps.t.ppf(0.975, dof) * se
            else:
                h = 0
            d.append(diff * 100); lo.append((diff - h) * 100); hi.append((diff + h) * 100)
        d, lo, hi = map(np.array, (d, lo, hi))
        ax.errorbar(ks + off, d, yerr=[d - lo, hi - d], color=COLOR[arch], marker=MARKER[arch],
                    capsize=3, label=LABEL[arch], linewidth=1.5, markeredgecolor="white")
    ax.set_xticks(ks)
    ax.set_xlabel("Demonstrations per command (k)")
    ax.set_ylabel("Hybrid − Pure paired accuracy (pp)")
    ax.set_title("Hybrid advantage over pure Gated DeltaNet")
    ax.legend(frameon=False)
    ax.text(0.0, -0.2, "Points: difference of seed means; bars: Welch 95% CI (5 vs 5 seeds), uncorrected.",
            transform=ax.transAxes, fontsize=7.5, color=INK2)
    save(fig, out, "fig3_hybrid_minus_pure")


def placement(ep, out):
    archs = [a for a in ARCH_ORDER if a in set(ep.arch)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
    for ax, col, title in ((axes[0], "icl_primary", "ICL paired accuracy (k = 1–8)"),
                           (axes[1], "icl_small_k", "ICL paired accuracy (k = 1–2)")):
        for i, arch in enumerate(archs):
            v = ep[ep.arch == arch][col].values * 100
            ax.bar(i, v.mean(), width=0.6, color=COLOR[arch], alpha=0.35, edgecolor="none")
            ax.errorbar(i, v.mean(), yerr=tci(v / 100) * 100, color=INK, capsize=4, linewidth=1.2)
            jitter = np.linspace(-0.12, 0.12, len(v))
            ax.scatter(i + jitter, v, color=COLOR[arch], marker=MARKER[arch], s=28,
                       edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_xticks(range(len(archs)))
        ax.set_xticklabels([SHORT[a] for a in archs])
        ax.set_ylabel("Paired accuracy (%)")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Attention placement comparison (bars: mean; whiskers: 95% t-CI; dots: individual seeds)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, out, "fig4_placement")


def loss_curves(ckpt_dir: Path, out):
    runs = {}
    for d in sorted(ckpt_dir.glob("*_s*")):
        if not (d / "train_log.jsonl").exists():
            continue
        meta = json.load(open(d / "metadata.json"))
        runs.setdefault(meta["arch"], []).append(read_jsonl(d / "train_log.jsonl"))
    if not runs:
        return
    panels = [("train_loss", "Training loss (all supervised result tokens)"),
              ("val_loss_all", "Validation loss (answer token)"),
              ("val_acc_icl", "Validation accuracy, ICL prompts"),
              ("val_acc_definition", "Validation accuracy, definition prompts")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (key, title) in zip(axes.flat, panels):
        for arch in [a for a in ARCH_ORDER if a in runs]:
            curves = []
            for log in runs[arch]:
                pts = [(x["step"], x[key]) for x in log if key in x]
                curves.append(dict(pts))
            steps = sorted(set.intersection(*[set(c) for c in curves]))
            M = np.array([[c[s] for s in steps] for c in curves])
            if key == "train_loss":  # smooth for readability
                kern = np.ones(10) / 10
                M = np.array([np.convolve(r, kern, mode="valid") for r in M])
                steps = steps[9:]
            m = M.mean(0)
            sd = M.std(0, ddof=1) if len(M) > 1 else np.zeros_like(m)
            if key.startswith("val_acc"):
                m, sd = m * 100, sd * 100
            ax.plot(steps, m, color=COLOR[arch], label=LABEL[arch], linewidth=1.6)
            ax.fill_between(steps, m - sd, m + sd, color=COLOR[arch], alpha=0.15, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Training step")
        if "loss" in key:
            ax.set_yscale("log"); ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f")); ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.set_ylabel("Cross-entropy (nats, log scale)")
        else:
            ax.set_ylabel("Accuracy (%)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Training dynamics (mean ± SD across seeds)", fontsize=11)
    fig.tight_layout()
    save(fig, out, "fig5_loss_curves")


def icl_vs_control(ep, out):
    archs = [a for a in ARCH_ORDER if a in set(ep.arch)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    ax = axes[0]
    conds = [("icl_primary", "ICL (demonstrations, k=1–8)"), ("definition_paired", "Control (explicit definition)")]
    width = 0.38
    for j, (col, name) in enumerate(conds):
        for i, arch in enumerate(archs):
            v = ep[ep.arch == arch][col].values * 100
            x = i + (j - 0.5) * width
            ax.bar(x, v.mean(), width=width - 0.04, color=COLOR[arch], alpha=0.9 if j == 0 else 0.35,
                   hatch=None if j == 0 else "///", edgecolor="white" if j == 0 else COLOR[arch], linewidth=0)
            ax.errorbar(x, v.mean(), yerr=tci(v / 100) * 100, color=INK, capsize=3, linewidth=1)
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels([SHORT[a] for a in archs])
    ax.set_ylabel("Paired accuracy (%)")
    ax.set_title("Paired accuracy: ICL vs direct-definition control")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#888", label=conds[0][1]),
                       Patch(facecolor="#ddd", hatch="///", edgecolor="#888", label=conds[1][1])],
              frameon=False, loc="upper left", fontsize=8, ncol=2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    if "pure" in archs:
        ax.axhline(0, color=INK2, linewidth=1)
        hybrids = [a for a in archs if a != "pure"]
        for i, arch in enumerate(hybrids):
            for j, (col, name) in enumerate(conds):
                a = ep[ep.arch == arch][col].values
                b = ep[ep.arch == "pure"][col].values
                diff = (a.mean() - b.mean()) * 100
                va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
                se = np.sqrt(va + vb)
                dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else 1
                h = sps.t.ppf(0.975, dof) * se * 100 if se > 0 else 0
                x = i + (j - 0.5) * 0.3
                ax.errorbar(x, diff, yerr=h, marker=MARKER[arch] if j == 0 else "o",
                            color=COLOR[arch], mfc=COLOR[arch] if j == 0 else "white", capsize=3,
                            linewidth=1.4, markersize=7)
        ax.set_xticks(range(len(hybrids)))
        ax.set_xticklabels([SHORT[a] for a in hybrids])
        ax.set_ylabel("Hybrid − Pure (pp)")
        ax.set_title("Hybrid advantage: ICL (filled) vs control (open)")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "fig6_icl_vs_control")


def by_operation(out, tdir):
    p = tdir / "accuracy_by_operation.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    ops = ["set", "add", "sub", "mul"]
    archs = [a for a in ARCH_ORDER if a in set(d.arch)]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    w = 0.8 / len(archs)
    for i, arch in enumerate(archs):
        g = d[d.arch == arch].set_index("op").reindex(ops)
        ax.bar(np.arange(4) + (i - (len(archs) - 1) / 2) * w, g["mean"] * 100, width=w - 0.02,
               color=COLOR[arch], label=LABEL[arch], yerr=g["std"] * 100, capsize=2,
               error_kw={"elinewidth": 0.8, "ecolor": INK2})
    ax.set_xticks(range(4))
    ax.set_xticklabels(ops)
    ax.set_xlabel("True operation of the target command")
    ax.set_ylabel("Individual accuracy (%)")
    ax.set_title("ICL accuracy by command meaning (k = 1–8; whiskers: SD across seeds)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig7_accuracy_by_operation")


def _val_curve(path: Path, key: str):
    log = read_jsonl(path)
    pts = {x["step"]: x[key] * 100 for x in log if key in x}   # resumed runs: latest entry per step wins
    xs = sorted(pts)
    return xs, [pts[x] for x in xs]


def per_seed_curves(ckpt_dir: Path, ep: pd.DataFrame, out: Path):
    """Fig 8 (added after seeing the bimodal outcome): one panel per training seed,
    one line per architecture; validation accuracy on definition prompts over training,
    annotated with the final test ICL paired accuracy."""
    seeds = sorted(ep.seed.unique())
    fig, axes = plt.subplots(1, len(seeds), figsize=(3.0 * len(seeds), 3.4), sharey=True)
    for ax, sd in zip(axes, seeds):
        for arch in ARCH_ORDER:
            f = ckpt_dir / f"{arch}_s{sd}" / "train_log.jsonl"
            if not f.exists():
                continue
            x, y = _val_curve(f, "val_acc_definition")
            ax.plot(x, y, color=COLOR[arch], label=LABEL[arch], linewidth=1.6)
        vals = ep[ep.seed == sd].set_index("arch")["icl_primary"] * 100
        ax.set_title(f"seed {sd}  (test ICL paired: {vals.min():.0f}–{vals.max():.0f}%)", fontsize=9)
        ax.set_xlabel("Training step")
        ax.set_xticks([0, 10000, 20000]); ax.set_xticklabels(["0", "10k", "20k"])
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Val. accuracy, definition prompts (%)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Per-seed training curves (one panel per training seed)",
                 fontsize=10, y=1.05)
    save(fig, out, "fig8_per_seed_curves")


def seedsplit(ckpt_dir: Path, out: Path):
    """Fig 9 (exploratory follow-up): pure model, data-stream seed x init seed."""
    runs = sorted(ckpt_dir.glob("pure_s*_d*/train_log.jsonl"))
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4))
    dcol = {0: "#2a78d6", 1: "#eb6834"}
    seen, ends = set(), []
    for f in runs:
        name = f.parent.name                     # pure_s10_d1
        init, d = int(name.split("_s")[1].split("_d")[0]), int(name.split("_d")[1])
        x, y = _val_curve(f, "val_acc_definition")
        lab = f"data stream {d}" if d not in seen else None
        seen.add(d)
        ax.plot(x, y, color=dcol.get(d, INK2), linewidth=1.6, label=lab)
        if x:
            ends.append([y[-1], x[-1], f"init {init}", dcol.get(d, INK2)])
    ends.sort()
    for i in range(1, len(ends)):              # stagger end labels so they do not overlap
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 3.5)
    for yl, xl, txt, c in ends:
        ax.annotate(txt, (xl, yl), fontsize=7, color=c, xytext=(4, 0),
                    textcoords="offset points", va="center", annotation_clip=False)
    ax.set_xlabel("Training step"); ax.set_ylabel("Val. accuracy, definition prompts (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Exploratory: pure GDN, data-stream seed x initialisation seed")
    ax.legend(frameon=False, loc="upper left")
    save(fig, out, "fig9_data_vs_init_seed")


def earlyinit(ckpt_dir: Path, seedsplit_dir: Path, out: Path):
    """Fig 10 (exploratory experiment A): pure model, data stream 1, early-layer (GDN layers 0-1
    decay + conv) initialisation transplanted from another seed. Body = all other parameters."""
    runs = sorted(ckpt_dir.glob("pure_b*_e*_d1*/train_log.jsonl"))
    if not runs:
        return
    import re as _re
    fig, ax = plt.subplots(figsize=(6.6, 4))
    bcol = {10: "#2a78d6", 11: "#eb6834", 12: "#1baa7d"}
    ends = []
    for b in (10, 11, 12):                      # originals from the data-vs-init follow-up, dashed
        f = seedsplit_dir / f"pure_s{b}_d1" / "train_log.jsonl"
        if f.exists():
            x, y = _val_curve(f, "val_acc_definition")
            ax.plot(x, y, color=bcol[b], linewidth=1.0, linestyle=":", alpha=0.8)
    seen = set()
    for f in runs:
        b, e = map(int, _re.match(r"pure_b(\d+)_e(\d+)", f.parent.name).groups())
        rerun = "rerun" in f.parent.name
        x, y = _val_curve(f, "val_acc_definition")
        seen.add(b)
        ax.plot(x, y, color=bcol.get(b, INK2), linewidth=1.6, linestyle="--" if rerun else "-")
        if x:
            ends.append([y[-1], x[-1], f"b{b}/e{e}" + (" rerun" if rerun else ""), bcol.get(b, INK2)])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 3.5)
    for yl, xl, txt, c in ends:
        ax.annotate(txt, (xl, yl), fontsize=7, color=c, xytext=(4, 0),
                    textcoords="offset points", va="center", annotation_clip=False)
    ax.set_xlabel("Training step"); ax.set_ylabel("Val. accuracy, definition prompts (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Exploratory A: early-layer init transplant (pure GDN, data stream 1)", fontsize=10)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=bcol[b], linewidth=1.6, label=f"body (other params) seed {b}")
               for b in sorted(seen)]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=8,
              title="solid: transplant, dashed: rerun, dotted: original", title_fontsize=7)
    save(fig, out, "fig10_early_init_transplant")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--ckpt-dir", default="checkpoints/main")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    tdir, out = Path(a.tables), Path(a.out)
    per_run = pd.read_csv(tdir / "per_run_condition_k.csv")
    ep = pd.read_csv(tdir / "seed_endpoints.csv")
    acc_vs_k(per_run, "paired_acc", "Paired accuracy vs number of demonstrations",
             "Paired accuracy (%)", out, "fig1_paired_accuracy_vs_k")
    acc_vs_k(per_run, "individual_acc", "Individual-answer accuracy vs number of demonstrations",
             "Individual accuracy (%)", out, "fig2_individual_accuracy_vs_k")
    diff_vs_k(per_run, out)
    placement(ep, out)
    loss_curves(Path(a.ckpt_dir), out)
    icl_vs_control(ep, out)
    by_operation(out, tdir)
    per_seed_curves(Path(a.ckpt_dir), ep, out)
    seedsplit(Path("checkpoints/seedsplit"), out)
    earlyinit(Path("checkpoints/earlyinit"), Path("checkpoints/seedsplit"), out)


if __name__ == "__main__":
    main()
