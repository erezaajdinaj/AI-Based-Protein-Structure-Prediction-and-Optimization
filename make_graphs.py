import argparse
import glob
import os
import sys

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KNOWN_OPTIMA = {
    "20_Benchmark": -9,
    "24_Benchmark": -9,
    "36_Benchmark": -14,
    "48_Benchmark": -23,
    "50_Benchmark": -21,
    "60_Benchmark": -36,
    "64_Benchmark": -42,
    "85_Benchmark": -53,
    "100a_Benchmark": -50,
    "100b_Benchmark": -48,
}
BASE_ORDER = ["Hill_Climbing", "Simulated_Annealing", "Genetic_Algorithm", "Beam_Search"]
SHORT = {"Hill_Climbing": "Hill", "Simulated_Annealing": "SA",
         "Genetic_Algorithm": "GA", "Beam_Search": "Beam"}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.titlesize": 11})


def parse_args():
    parser = argparse.ArgumentParser(description="Build figures and tables from experiment CSV files")
    parser.add_argument("folder", help="folder with the CSV files, for example results/baseline")
    parser.add_argument("--results", default="experiment_results*.csv", help="pattern of the results file")
    parser.add_argument("--history", default="convergence_history*.csv", help="pattern of the history file")
    parser.add_argument("--merge", nargs="*", default=[],
                        help="other result folders added to the comparison, for example results/baseline")
    parser.add_argument("--only", default=None,
                        help="comma separated Algorithm names to keep, for example Genetic_Algorithm,GA_v2")
    parser.add_argument("--show", action="store_true", help="also open the figures in windows")
    return parser.parse_args()


def find_file(folder, pattern):
    matches = glob.glob(os.path.join(folder, pattern))
    if not matches:
        raise SystemExit(f"No file matching '{pattern}' in '{folder}'")
    newest = max(matches, key=os.path.getmtime)
    print("Using", newest)
    return newest


def short(name):
    return SHORT.get(name, name)


def main():
    args = parse_args()
    folder = args.folder
    fig_dir = os.path.join(folder, "figures")
    table_dir = os.path.join(folder, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    df = pd.read_csv(find_file(folder, args.results))
    hist = pd.read_csv(find_file(folder, args.history))
    for other in args.merge:
        df = pd.concat([df, pd.read_csv(find_file(other, args.results))], ignore_index=True)
        hist = pd.concat([hist, pd.read_csv(find_file(other, args.history))], ignore_index=True)
    if args.only:
        keep = args.only.split(",")
        df = df[df["Algorithm"].isin(keep)].copy()
        hist = hist[hist["Algorithm"].isin(keep)].copy()
    print(len(df), "result rows,", len(hist), "history rows")

    time_limit = float(round(df["Time_Sec"].median()))
    present = list(df["Algorithm"].unique())
    algos = [a for a in BASE_ORDER if a in present] + sorted(a for a in present if a not in BASE_ORDER)
    cmap = plt.get_cmap("tab10")
    colors = {a: cmap(i % 10) for i, a in enumerate(algos)}
    markers = {a: MARKERS[i % len(MARKERS)] for i, a in enumerate(algos)}

    df["Optimum"] = df["Sequence_Name"].map(KNOWN_OPTIMA)
    df["Gap"] = df["Best_Energy"] - df["Optimum"]
    df["Hit"] = np.where(df["Optimum"].notna(), (df["Gap"] == 0).astype(float), np.nan)
    df["Invalid_Pct"] = df["Invalid"] / df["Explored"].where(df["Explored"] > 0) * 100
    if "Pruned" in df:
        df["Pruning_Percent"] = df["Pruning_Percent"].where(df["Pruned"] > 0)

    last_improvement = (hist.groupby(["Sequence_Name", "Algorithm", "Seed"])["Time_Sec"]
                        .max().rename("Time_To_Best").reset_index())
    df = df.merge(last_improvement, on=["Sequence_Name", "Algorithm", "Seed"], how="left")

    seqs = (df.drop_duplicates("Sequence_Name")
              .sort_values(["Length", "Sequence_Name"])["Sequence_Name"].tolist())
    n_seeds = int(df.groupby(["Sequence_Name", "Algorithm"]).size().max())

    def finish(fig, name):
        fig.tight_layout()
        path = os.path.join(fig_dir, name)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print("saved", path)
        if args.show:
            plt.show()
        plt.close(fig)

    def line_plot(column, ylabel, title, filename, logy=False, error=True):
        mean = df.pivot_table(index="Length", columns="Algorithm", values=column, aggfunc="mean")
        std = df.pivot_table(index="Length", columns="Algorithm", values=column, aggfunc="std")
        fig, ax = plt.subplots(figsize=(8, 5))
        for algo in algos:
            if algo not in mean.columns or mean[algo].isna().all():
                continue
            yerr = std[algo].values if error else None
            ax.errorbar(mean.index, mean[algo].values, yerr=yerr, marker=markers[algo],
                        color=colors[algo], capsize=3, label=algo)
        ax.set_xlabel("Sequence length")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(list(mean.index))
        ax.legend()
        finish(fig, filename)

    def grouped_bar(pivot, error, labels, ylabel, title, filename):
        fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(pivot)), 5))
        count = len(algos)
        width = 0.8 / count
        x = np.arange(len(pivot))
        for i, algo in enumerate(algos):
            if algo not in pivot.columns:
                continue
            err = error[algo].values if error is not None else None
            ax.bar(x + (i - (count - 1) / 2) * width, pivot[algo].values, width,
                   yerr=err, capsize=2, label=algo, color=colors[algo])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        finish(fig, filename)

    line_plot("Best_Energy", "Best energy found (mean, error bar = std over seeds)",
              f"Solution quality vs sequence length ({n_seeds} seeds per point)",
              "01_energy_vs_length.png")

    bench = df[df["Optimum"].notna()]
    if not bench.empty:
        bench_seqs = [s for s in seqs if s in KNOWN_OPTIMA]
        gap_mean = bench.pivot_table(index="Sequence_Name", columns="Algorithm", values="Gap", aggfunc="mean").reindex(bench_seqs)
        gap_std = bench.pivot_table(index="Sequence_Name", columns="Algorithm", values="Gap", aggfunc="std").reindex(bench_seqs)
        hits = bench.pivot_table(index="Sequence_Name", columns="Algorithm", values="Hit", aggfunc="sum").reindex(bench_seqs)
        labels = [f"n={s.replace('_Benchmark', '')}\nbest known {KNOWN_OPTIMA[s]}" for s in bench_seqs]
        grouped_bar(gap_mean, gap_std, labels, "Gap to best known energy (0 = optimum)",
                    "Gap to the best known energy on benchmark sequences", "02_gap_to_optimum.png")
        grouped_bar(hits, None, labels, f"Runs reaching the best known energy (out of {n_seeds})",
                    "Optimum hits on benchmark sequences", "03_optimum_hits.png")
        gap_mean.round(2).to_csv(os.path.join(table_dir, "pivot_mean_gap.csv"))
        hits.to_csv(os.path.join(table_dir, "pivot_optimum_hits.csv"))

    columns = 3 if len(seqs) > 4 else 2
    rows = int(np.ceil(len(seqs) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.9 * rows), squeeze=False)
    grid = np.linspace(0, time_limit, 400)
    for ax, seq in zip(axes.flat, seqs):
        for algo in algos:
            curves = []
            runs = hist[(hist["Sequence_Name"] == seq) & (hist["Algorithm"] == algo)]
            for _, run in runs.groupby("Seed"):
                run = run.sort_values("Time_Sec")
                index = np.searchsorted(run["Time_Sec"].values, grid, side="right") - 1
                curves.append(run["Best_Energy"].values[np.clip(index, 0, None)])
            if not curves:
                continue
            curves = np.array(curves)
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            ax.plot(grid, mean, color=colors[algo], label=short(algo))
            ax.fill_between(grid, mean - std, mean + std, color=colors[algo], alpha=0.12)
        title = f"{seq}"
        if seq in KNOWN_OPTIMA:
            ax.axhline(KNOWN_OPTIMA[seq], color="black", linestyle="--", linewidth=1)
            title += f" (best known {KNOWN_OPTIMA[seq]})"
        ax.set_title(title)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlim(0, time_limit)
        ax.set_xlabel("Time (s, symlog)")
        ax.set_ylabel("Best energy so far")
    for ax in list(axes.flat)[len(seqs):]:
        ax.set_visible(False)
    axes.flat[0].legend()
    fig.suptitle("Convergence: mean best energy over time (shaded = std over seeds)")
    finish(fig, "04_convergence.png")

    line_plot("Time_To_Best", "Time of the last improvement (s)",
              "Convergence speed: time until the final best energy was found",
              "05_time_to_best.png")

    line_plot("Memory_MB", "Peak memory (MB, log scale)",
              "Memory consumption vs sequence length (tracemalloc)",
              "06_memory_vs_length.png", logy=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    explored = df.pivot_table(index="Length", columns="Algorithm", values="Explored", aggfunc="mean")
    invalid = df.pivot_table(index="Length", columns="Algorithm", values="Invalid_Pct", aggfunc="mean")
    for algo in algos:
        ax1.plot(explored.index, explored[algo].values, marker=markers[algo], color=colors[algo], label=algo)
        ax2.plot(invalid.index, invalid[algo].values, marker=markers[algo], color=colors[algo], label=algo)
    ax1.set_xlabel("Sequence length")
    ax1.set_ylabel("Configurations explored (mean per run)")
    ax1.set_title("Configurations explored in the same time budget")
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("Invalid configurations (% of explored)")
    ax2.set_title("Structural validity: share of rejected candidates")
    ax1.legend()
    finish(fig, "07_explored_and_invalid.png")

    pruned_algos = [a for a in algos if df.loc[df["Algorithm"] == a, "Pruning_Percent"].notna().any()]
    if pruned_algos:
        pruning = df.pivot_table(index="Length", columns="Algorithm", values="Pruning_Percent", aggfunc="mean")
        grouped_bar(pruning[pruned_algos], None, [f"n={n}" for n in pruning.index],
                    "Valid candidates discarded by pruning (%)",
                    "Search-space reduction: share of valid candidates discarded",
                    "08_pruning.png")

    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.9 * rows), squeeze=False)
    for ax, seq in zip(axes.flat, seqs):
        data = [df[(df["Sequence_Name"] == seq) & (df["Algorithm"] == a)]["Best_Energy"].values for a in algos]
        box = ax.boxplot(data, patch_artist=True)
        for patch, algo in zip(box["boxes"], algos):
            patch.set_facecolor(colors[algo])
            patch.set_alpha(0.5)
        ax.set_xticklabels([short(a) for a in algos])
        title = f"{seq}"
        if seq in KNOWN_OPTIMA:
            ax.axhline(KNOWN_OPTIMA[seq], color="black", linestyle="--", linewidth=1)
            title += f" (best known {KNOWN_OPTIMA[seq]})"
        ax.set_title(title)
        ax.set_ylabel("Best energy")
    for ax in list(axes.flat)[len(seqs):]:
        ax.set_visible(False)
    fig.suptitle("Variability across seeds")
    finish(fig, "09_energy_boxplots.png")

    if "Beam_Width" in df:
        width = pd.to_numeric(df["Beam_Width"], errors="coerce")
        if width.notna().any() and (width > 0).any():
            df["Beam_Width_Num"] = width.where(width > 0)
            line_plot("Beam_Width_Num", "Final beam width reached (log scale)",
                      "Beam width reached within the time budget", "10_beam_width.png", logy=True)

    summary = df.groupby(["Sequence_Name", "Length", "Algorithm"]).agg(
        Runs=("Seed", "count"),
        Mean_Energy=("Best_Energy", "mean"),
        Std_Energy=("Best_Energy", "std"),
        Min_Energy=("Best_Energy", "min"),
        Max_Energy=("Best_Energy", "max"),
        Best_Known=("Optimum", "first"),
        Mean_Gap=("Gap", "mean"),
        Optimum_Hits=("Hit", lambda s: s.sum() if s.notna().any() else np.nan),
        Mean_Time_Sec=("Time_Sec", "mean"),
        Time_To_Best=("Time_To_Best", "mean"),
        Mean_Memory_MB=("Memory_MB", "mean"),
        Mean_Explored=("Explored", "mean"),
        Mean_Valid=("Valid", "mean"),
        Mean_Invalid=("Invalid", "mean"),
        Invalid_Pct=("Invalid_Pct", "mean"),
        Pruning_Pct=("Pruning_Percent", "mean"),
    ).round(2).reset_index().sort_values(["Length", "Sequence_Name", "Algorithm"])
    summary.to_csv(os.path.join(table_dir, "summary_table.csv"), index=False)

    energy_pivot = df.pivot_table(index="Sequence_Name", columns="Algorithm", values="Best_Energy", aggfunc="mean")
    energy_pivot = energy_pivot.reindex(seqs)[algos].round(2)
    energy_pivot.to_csv(os.path.join(table_dir, "pivot_mean_energy.csv"))

    overall = bench.groupby("Algorithm").agg(
        Mean_Gap=("Gap", "mean"), Optimum_Hits=("Hit", "sum"), Runs=("Hit", "count"))
    overall["Mean_Relative_Gap_Pct"] = (bench.assign(rel=bench["Gap"] / bench["Optimum"].abs() * 100)
                                        .groupby("Algorithm")["rel"].mean())
    overall = overall.reindex(algos).round(2)
    overall.to_csv(os.path.join(table_dir, "overall_ranking.csv"))

    print("\nMean energy per sequence:")
    print(energy_pivot.to_string())
    print("\nOverall (benchmark sequences only):")
    print(overall.to_string())
    print("\nFigures:", fig_dir)
    print("Tables:", table_dir)


if __name__ == "__main__":
    main()