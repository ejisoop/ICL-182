"""Load-scaling pilot analysis (IMPLEMENTATION_PLAN.md §11).

    python scripts/analyze_recall.py [--raw-dir results/raw_recall] [--out results/recall]

Per run: query accuracy by (format, N).  Per architecture: mean (and per-seed values) over seeds.
Hybrid - pure differences per N, paired by seed.  Writes tables/*.md|csv, figures/*.png|pdf and
summary.json, and prints the pre-registered pilot readout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ARCHS = ["pure", "hybrid_early", "hybrid_middle", "hybrid_late"]
LABEL = {"pure": "Pure GDN", "hybrid_early": "Hybrid, attn @ L2 (early)",
         "hybrid_middle": "Hybrid, attn @ L4 (middle)", "hybrid_late": "Hybrid, attn @ L6 (late)"}
COLOR = {"pure": "#2a78d6", "hybrid_early": "#eb6834", "hybrid_middle": "#1baa7d", "hybrid_late": "#e8a317"}
FMT_NAME = {"icl": "demonstrations (D c x = y)", "definition": "stated meanings (M c v)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw_recall")
    ap.add_argument("--out", default="results/recall")
    a = ap.parse_args()
    out = Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(Path(a.raw_dir).glob("*.csv"))])
    run = df.groupby(["arch", "seed", "format", "n"]).correct.mean().reset_index()
    run.to_csv(out / "tables" / "per_run_accuracy.csv", index=False)
    archs = [x for x in ARCHS if x in set(run.arch)]
    ns = sorted(run.n.unique())

    lines, summary = [], {"per_arch": {}, "diff_vs_pure": {}}
    for fmt in ("icl", "definition"):
        lines += [f"\n### Query accuracy (%), {FMT_NAME[fmt]}: mean over seeds [per-seed values]\n",
                  "| arch | " + " | ".join(f"N={n}" for n in ns) + " |", "|---" * (len(ns) + 1) + "|"]
        for arch in archs:
            cells = []
            for n in ns:
                v = run[(run.arch == arch) & (run.format == fmt) & (run.n == n)].sort_values("seed").correct * 100
                cells.append(f"{v.mean():.1f} [{', '.join(f'{x:.0f}' for x in v)}]")
                summary["per_arch"][f"{arch}|{fmt}|{n}"] = [float(x) for x in v]
            lines.append(f"| {arch} | " + " | ".join(cells) + " |")
        if "pure" in archs:
            lines += [f"\nHybrid - pure (pp), {FMT_NAME[fmt]}: mean over seeds [per-seed differences]\n",
                      "| hybrid | " + " | ".join(f"N={n}" for n in ns) + " |", "|---" * (len(ns) + 1) + "|"]
            for arch in [x for x in archs if x != "pure"]:
                cells = []
                for n in ns:
                    h = run[(run.arch == arch) & (run.format == fmt) & (run.n == n)].set_index("seed").correct
                    p = run[(run.arch == "pure") & (run.format == fmt) & (run.n == n)].set_index("seed").correct
                    d = ((h - p).dropna() * 100)
                    cells.append(f"{d.mean():+.1f} [{', '.join(f'{x:+.0f}' for x in d)}]")
                    summary["diff_vs_pure"][f"{arch}|{fmt}|{n}"] = [float(x) for x in d]
                lines.append(f"| {arch} | " + " | ".join(cells) + " |")
    # accuracy by position of the queried binding (recency), largest N, ICL format
    nmax = max(ns)
    sub = df[(df.n == nmax) & (df.format == "icl")].copy()
    sub["rank_bin"] = pd.cut(sub["rank"], bins=4, labels=["first quarter", "2nd", "3rd", "last quarter"])
    rb = (sub.groupby(["arch", "rank_bin"], observed=True).correct.mean().unstack() * 100).round(1)
    lines += [f"\n### N={nmax}, demonstrations: accuracy (%) by position of the queried binding in the prompt\n",
              "| arch | " + " | ".join(map(str, rb.columns)) + " |", "|---" * (len(rb.columns) + 1) + "|"]
    lines += [f"| {arch} | " + " | ".join(f"{v:.1f}" for v in rb.loc[arch]) + " |" for arch in rb.index]
    (out / "tables" / "recall_summary.md").write_text("\n".join(lines) + "\n")
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, fmt in zip(axes, ("icl", "definition")):
        for arch in archs:
            g = run[(run.arch == arch) & (run.format == fmt)]
            for sd, gs in g.groupby("seed"):
                ax.plot(gs.n, gs.correct * 100, color=COLOR[arch], alpha=0.35, linewidth=0.8)
            m = g.groupby("n").correct.mean() * 100
            ax.plot(m.index, m.values, color=COLOR[arch], marker="o", linewidth=2, label=LABEL[arch])
        ax.axhline(100 / 13, color="#888", linestyle=":", linewidth=1)
        ax.text(ns[0], 100 / 13 + 1.5, "chance", fontsize=7, color="#888")
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Number of bound commands in the prompt (N)"); ax.set_title(f"Meanings via {FMT_NAME[fmt]}")
        ax.set_ylim(0, 102); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Query accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(0.5, -0.02, "Thick: mean over seeds; thin: individual seeds. Fixed test set, 250 prompts x 8 queries per cell.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"recall_accuracy_vs_load.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {out}/figures/recall_accuracy_vs_load")


if __name__ == "__main__":
    main()
