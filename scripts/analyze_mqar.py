"""Summarize completed MQAR pilot runs from their per-query test CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOADS = (4, 8, 16, 32, 64, 128, 256)
HIGH_LOADS = (128, 256)
SEEDS = (0, 1)
ARMS = {
    64: ("transformer", "pure", "hybrid_early", "hybrid_middle", "hybrid_late"),
    32: ("pure", "hybrid_middle"),
}
COLORS = {
    "transformer": "#443783", "pure": "#222222", "hybrid_early": "#007c91",
    "hybrid_middle": "#b54519", "hybrid_late": "#4b732e",
}
FIELDS = ["gdn_head_dim", "arch", "seed"]
QUERY_FIELDS = ["n", "id", "query_idx", "rank", "answer"]


def load_queries(raw_dir, manifest_path):
    paths = sorted(raw_dir.glob("*.csv"))
    if not paths:
        raise ValueError(f"no raw query CSVs in {raw_dir}")
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    groups = df.groupby(FIELDS, sort=False, dropna=False)
    expected = {(dim, arch, seed) for dim, arches in ARMS.items() for arch in arches for seed in SEEDS}
    if set(groups.groups) != expected:
        raise ValueError(f"expected all 14 pilot runs, missing {expected - set(groups.groups)}, "
                         f"unexpected {set(groups.groups) - expected}")
    manifest = json.loads(manifest_path.read_text())
    expected_counts = {int(n): count for n, count in manifest["test"]["queries_per_n"].items()}
    reference = None
    for run, queries in groups:
        if queries.groupby("n").size().to_dict() != expected_counts:
            raise ValueError(f"query counts do not match the fixed test manifest for {run}")
        if queries.duplicated(["id", "query_idx"]).any():
            raise ValueError(f"duplicate query in {run}")
        if ((queries["rank"] < 0) | (queries["rank"] >= queries["n"])).any():
            raise ValueError(f"invalid binding rank in {run}")
        if not queries["correct"].eq(queries["answer"].eq(queries["pred"]).astype(int)).all():
            raise ValueError(f"inconsistent correctness labels in {run}")
        sample = queries[QUERY_FIELDS].sort_values(QUERY_FIELDS).reset_index(drop=True)
        if reference is not None and not sample.equals(reference):
            raise ValueError(f"test query identities differ for {run}")
        reference = sample
    return df


def accuracy_by(df, fields):
    summary = df.groupby(fields, as_index=False).agg(correct=("correct", "sum"), queries=("correct", "size"))
    return summary.assign(accuracy=summary["correct"] / summary["queries"])


def hybrid_differences(accuracy):
    matching = ["gdn_head_dim", "seed", "n"]
    pure = accuracy.loc[accuracy["arch"] == "pure", matching + ["accuracy"]]
    pure = pure.rename(columns={"accuracy": "pure_accuracy"})
    hybrids = accuracy[accuracy["arch"].str.startswith("hybrid_")]
    diff = hybrids.merge(pure, on=matching, validate="many_to_one")
    return diff.assign(gap_pp=100 * (diff["accuracy"] - diff["pure_accuracy"]))


def decision(accuracy, differences):
    acc = accuracy.set_index(FIELDS + ["n"])["accuracy"]
    failed = {tuple(run) for run in acc.xs(8, level="n").loc[lambda values: values < 0.9].index}
    control = [seed for seed in SEEDS if acc.loc[(64, "transformer", seed, 128)] < 0.9]
    rules = ["R1"] if failed else []
    result = {"rules": rules, "failed_runs": [list(run) for run in sorted(failed)],
              "broken_control_seeds": control, "qualifying_comparisons": []}
    if control:
        rules.append("R2")
        return result
    if any((64, arch, seed) in failed for arch in ("pure", "transformer") for seed in SEEDS):
        return {**result, "needs_clarification": "An ineligible run is required to establish R2 or R3."}

    low_load = all(acc.loc[(64, "pure", seed, 256)] + 1e-12 >= 0.9 * acc.loc[(64, "pure", seed, 8)]
                   for seed in SEEDS)
    eligible = differences[differences["n"].isin(HIGH_LOADS)]
    if low_load:
        rules.append("R3")
        eligible = eligible[eligible["gdn_head_dim"] == 32]
    result["primary_arm"] = "B" if low_load else "A"

    separated, one_seed, unpaired_gap = [], [], False
    complete_comparisons = 0
    for (dim, arch, n), rows in eligible.groupby(["gdn_head_dim", "arch", "n"]):
        rows = rows[~rows["seed"].map(lambda seed: (dim, arch, seed) in failed or (dim, "pure", seed) in failed)]
        hits = rows[rows["gap_pp"] >= 10 - 1e-10]
        if len(rows) < len(SEEDS):
            unpaired_gap |= bool(len(hits))
            continue
        complete_comparisons += 1
        comparison = {"gdn_head_dim": int(dim), "arch": arch, "n": int(n),
                      "seeds": [int(seed) for seed in hits["seed"]]}
        if len(hits) == len(SEEDS):
            separated.append(comparison)
        elif len(hits) == 1:
            one_seed.append(comparison)
    if separated:
        rules.append("R4")
        result["qualifying_comparisons"] = separated
    elif one_seed:
        rules.append("R5")
        result["qualifying_comparisons"] = one_seed
    elif unpaired_gap or not complete_comparisons:
        result["needs_clarification"] = "Failed runs prevent an eligible paired decision."
    elif not low_load:
        rules.append("R6")
    return result


def save_figure(fig, out, name):
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{name}.{suffix}", dpi=160)
    plt.close(fig)


def load_plot(data, out, difference=False):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, layout="constrained")
    for row, dim in enumerate(ARMS):
        for col, seed in enumerate(SEEDS):
            ax = axes[row, col]
            panel = data[(data["gdn_head_dim"] == dim) & (data["seed"] == seed)]
            for arch, points in panel.groupby("arch", sort=False):
                y = points["gap_pp"] if difference else 100 * points["accuracy"]
                ax.plot(points["n"], y, marker="o", color=COLORS[arch], label=arch)
            ax.set_title(f"Width {dim}, seed {seed}")
            ax.set_xscale("log", base=2)
            ax.set_xticks(LOADS, labels=LOADS)
            ax.grid(axis="y", alpha=0.2)
            if difference:
                ax.axhline(0, color="0.6", linewidth=0.8)
                ax.axhline(10, color="0.45", linewidth=0.8, linestyle="--")
            else:
                ax.set_ylim(0, 102)
                ax.axhline(100 / 64, color="0.45", linewidth=0.8, linestyle="--")
            if col == 1:
                ax.legend(frameon=False, fontsize=8)
    fig.supxlabel("Number of bindings (N)")
    fig.supylabel("Hybrid − pure (percentage points)" if difference else "Query accuracy (%)")
    save_figure(fig, out, "hybrid_minus_pure" if difference else "accuracy_by_load")


def recency_plot(bins, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True, layout="constrained")
    for row, dim in enumerate(ARMS):
        for col, n in enumerate(HIGH_LOADS):
            ax = axes[row, col]
            panel = bins[(bins["gdn_head_dim"] == dim) & (bins["n"] == n)]
            for (arch, seed), points in panel.groupby(["arch", "seed"], sort=False):
                ax.plot(points["rank_midpoint"], 100 * points["accuracy"], color=COLORS[arch],
                        linestyle="-" if seed == 0 else "--", label=f"{arch}, seed {seed}")
            ax.set_title(f"Width {dim}, N = {n}")
            ax.set_xlim(0, n - 1)
            ax.set_ylim(0, 102)
            ax.grid(axis="y", alpha=0.2)
            if col == 1:
                ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.supxlabel("Binding rank (oldest to newest)")
    fig.supylabel("Query accuracy (%)")
    save_figure(fig, out, "recency")


def analyze_validation(checkpoints, out):
    records = []
    for path in sorted(checkpoints.glob("*/metadata.json")):
        meta = json.loads(path.read_text())
        log = path.with_name("train_log.jsonl")
        if not log.exists():
            continue
        keys = [f"val_acc_n{n}" for n in LOADS]
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        evaluations = [row for row in rows if all(key in row for key in keys)]
        if not evaluations:
            continue
        final = max(evaluations[::-1], key=lambda row: row["step"])
        for n, key in zip(LOADS, keys):
            records.append({"arch": meta["arch"], "seed": meta["seed"], "step": final["step"],
                            "completed": meta.get("final_step") == final["step"],
                            "n_params": meta["n_params"], "n": n, "accuracy": final[key]})
    if not records:
        raise ValueError(f"no validation measurements in {checkpoints}")
    data = pd.DataFrame(records)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "tables" / "validation_accuracy.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.1), layout="constrained")
    for (arch, seed, step), points in data.groupby(["arch", "seed", "step"], sort=False):
        ax.plot(points["n"], 100 * points["accuracy"], marker="o", color=COLORS[arch],
                label=f"{arch.replace('_', ' ')}, seed {seed}, step {step:,}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(LOADS, labels=LOADS)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Number of bindings (N)")
    ax.set_ylabel("Validation query accuracy (%)")
    ax.axhline(90, color="0.55", linewidth=0.8, linestyle="--", label="90% control threshold")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, fontsize=9)
    save_figure(fig, out / "figures", "validation_accuracy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw_mqar"))
    parser.add_argument("--out", type=Path, default=Path("results/mqar"))
    parser.add_argument("--manifest", type=Path, default=Path("data/mqar/data_manifest.json"))
    parser.add_argument("--validation-dir", type=Path, help="summarize validation metrics from training logs")
    args = parser.parse_args()
    if args.validation_dir:
        analyze_validation(args.validation_dir, args.out)
        return
    data = load_queries(args.raw_dir, args.manifest)
    accuracy = accuracy_by(data, FIELDS + ["n"])
    differences = hybrid_differences(accuracy)
    high = data[data["n"].isin(HIGH_LOADS)].copy()
    high["rank_bin"] = high["rank"] * 16 // high["n"]
    bins = accuracy_by(high, FIELDS + ["n", "rank_bin"])
    bins["rank_midpoint"] = (bins["rank_bin"] + 0.5) * bins["n"] / 16 - 0.5
    pure = accuracy[accuracy["arch"] == "pure"]
    degradation = []
    for (dim, seed), series in pure.groupby(["gdn_head_dim", "seed"]):
        curve = series.set_index("n")["accuracy"]
        below = curve[(curve.index > 8) & (curve + 1e-12 < 0.9 * curve.loc[8])]
        degradation.append({"gdn_head_dim": dim, "seed": seed,
                            "first_n_below_90pct_of_n8": int(below.index.min()) if len(below) else ""})

    tables = args.out / "tables"
    figures = args.out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name, frame in {"accuracy_by_n": accuracy, "hybrid_minus_pure": differences,
                        "accuracy_by_rank": accuracy_by(high, FIELDS + ["n", "rank"]),
                        "recency_bins": bins, "pure_degradation_point": pd.DataFrame(degradation)}.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
    outcome = json.dumps(decision(accuracy, differences), indent=2)
    (args.out / "decision.json").write_text(outcome + "\n")
    load_plot(accuracy, figures)
    load_plot(differences, figures, difference=True)
    recency_plot(bins, figures)
    print(outcome)


if __name__ == "__main__":
    main()
