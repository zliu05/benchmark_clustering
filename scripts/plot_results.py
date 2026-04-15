#!/usr/bin/env python3
"""
Aggregate benchmark results and generate comparison plots.

Reads all JSON result files from results/ and produces:
  - benchmark_summary.csv
  - benchmark_runtime.png        (grouped bar chart by implementation)
  - benchmark_scalability.png    (cells vs. time, log-log)
  - benchmark_modularity.png     (modularity comparison)
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

IMPL_ORDER = [
    "scanpy/leidenalg",
    "seurat/leidenbase",
    "seurat/igraph",
    "gve-leiden/openmp",
]

IMPL_COLORS = {
    "scanpy/leidenalg": "#1f77b4",
    "seurat/leidenbase": "#ff7f0e",
    "seurat/igraph": "#ffbb78",
    "gve-leiden/openmp": "#2ca02c",
}

DATASET_ORDER = ["pbmc3k", "tabula_sapiens", "pbmc10k"]
DATASET_LABELS = {
    "pbmc3k": "PBMC 3k",
    "tabula_sapiens": "Tabula Sapiens",
    "pbmc10k": "PBMC 10k",
}


def get_dataset_labels_with_counts(df):
    """Generate dataset labels with actual cell counts from data."""
    labels = {}
    for ds in DATASET_ORDER:
        ds_data = df[df["dataset"] == ds]
        if not ds_data.empty:
            n_cells = ds_data.iloc[0]["n_cells"]
            if n_cells > 0:
                labels[ds] = f"{DATASET_LABELS[ds]}\n({n_cells:,} cells)"
            else:
                labels[ds] = DATASET_LABELS[ds]
        else:
            labels[ds] = DATASET_LABELS[ds]
    return labels


def load_results():
    """Load all individual JSON result files into a DataFrame."""
    records = []

    json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    # Skip old aggregated result files (they're arrays, not individual results)
    summary_files = {
        "scanpy_results.json", "seurat_results.json", "gveleiden_results.json",
        "scanpy_results copy.json", "seurat_results copy.json", "gveleiden_results copy.json",
        "scanpy_results-04142026.json", "seurat_results-04142026.json",
        "seurat_results_03182026.json", "seurat_results_04152026.json",
        "gveleiden_results_04142026.json",
    }

    for fpath in sorted(json_files):
        fname = os.path.basename(fpath)
        if fname in summary_files:
            continue

        with open(fpath) as f:
            data = json.load(f)

        # Skip if data is a list (old format) instead of dict
        if isinstance(data, list):
            print(f"  [skip] {fname} (old array format)")
            continue

        tool = data.get("tool", "unknown")
        impl = data.get("implementation", "unknown")

        if tool == "gve-leiden":
            impl_label = "gve-leiden/openmp"
            n_threads = data.get("n_threads", 1)
        else:
            impl_label = f"{tool}/{impl}"
            n_threads = 1

        record = {
            "tool": tool,
            "implementation": impl,
            "impl_label": impl_label,
            "dataset": data.get("dataset", "unknown"),
            "n_cells": data.get("n_cells", 0),
            "resolution": data.get("resolution", 1.0),
            "n_threads": n_threads,
            "mean_time": data.get("mean_time", 0),
            "std_time": data.get("std_time", 0),
            "min_time": data.get("min_time", 0),
            "n_clusters": data.get("n_clusters", 0),
            "modularity": data.get("modularity"),
            "peak_memory_mb": data.get("peak_memory_mb"),
            "individual_times": data.get("times", []),
            "source_file": fname,
        }
        records.append(record)

    if not records:
        print("No individual result files found in", RESULTS_DIR)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    
    # Remove scanpy/igraph results
    df = df[df["impl_label"] != "scanpy/igraph"].copy()
    
    return df


def plot_runtime_comparison(df, resolution=1.0):
    """Grouped bar chart comparing runtime across implementations and datasets."""
    sub = df[(df["resolution"] == resolution)].copy()
    if sub.empty:
        sub = df.copy()

    # For gve-leiden, pick the run with most threads
    gve = sub[sub["tool"] == "gve-leiden"]
    if not gve.empty:
        for ds in gve["dataset"].unique():
            ds_gve = gve[gve["dataset"] == ds]
            best_idx = ds_gve["mean_time"].idxmin()
            drop_idx = ds_gve.index[ds_gve.index != best_idx]
            sub = sub.drop(drop_idx)

    datasets_present = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    impls_present = [i for i in IMPL_ORDER if i in sub["impl_label"].unique()]

    if not datasets_present or not impls_present:
        print("Not enough data to plot runtime comparison.")
        return

    dataset_labels = get_dataset_labels_with_counts(df)
    
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets_present))
    width = 0.15
    n_impls = len(impls_present)
    offsets = np.arange(n_impls) - (n_impls - 1) / 2

    for i, impl in enumerate(impls_present):
        means = []
        stds = []
        for ds in datasets_present:
            row = sub[(sub["dataset"] == ds) & (sub["impl_label"] == impl)]
            if row.empty:
                means.append(0)
                stds.append(0)
            else:
                means.append(row.iloc[0]["mean_time"])
                stds.append(row.iloc[0]["std_time"])

        color = IMPL_COLORS.get(impl, "#999999")
        bars = ax.bar(
            x + offsets[i] * width,
            means,
            width,
            yerr=stds,
            label=impl,
            color=color,
            edgecolor="white",
            capsize=3,
        )

        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.2f}s",
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title(f"Leiden Clustering Runtime Comparison (resolution={resolution})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([dataset_labels.get(d, d) for d in datasets_present])
    ax.legend(loc="upper left", fontsize=9)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "benchmark_runtime.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_scalability(df, resolution=1.0):
    """Log-log plot of cells vs. runtime for each implementation."""
    sub = df[df["resolution"] == resolution].copy() if resolution in df["resolution"].values else df.copy()

    # For gve-leiden, pick best thread count per dataset
    gve = sub[sub["tool"] == "gve-leiden"]
    if not gve.empty:
        for ds in gve["dataset"].unique():
            ds_gve = gve[gve["dataset"] == ds]
            best_idx = ds_gve["mean_time"].idxmin()
            drop_idx = ds_gve.index[ds_gve.index != best_idx]
            sub = sub.drop(drop_idx)

    impls_present = [i for i in IMPL_ORDER if i in sub["impl_label"].unique()]
    if not impls_present:
        print("Not enough data for scalability plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for impl in impls_present:
        impl_data = sub[sub["impl_label"] == impl].sort_values("n_cells")
        if impl_data.empty or impl_data["n_cells"].max() == 0:
            continue
        color = IMPL_COLORS.get(impl, "#999999")
        ax.errorbar(
            impl_data["n_cells"],
            impl_data["mean_time"],
            yerr=impl_data["std_time"],
            label=impl,
            color=color,
            marker="o",
            linewidth=2,
            capsize=4,
        )

    ax.set_xlabel("Number of Cells", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("Scalability: Leiden Runtime vs. Dataset Size", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "benchmark_scalability.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_thread_scaling(df):
    """For GVE-Leiden: runtime vs thread count (strong scaling)."""
    gve = df[df["tool"] == "gve-leiden"].copy()
    if gve.empty:
        return

    datasets = gve["dataset"].unique()
    if len(datasets) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D"]

    for i, ds in enumerate(datasets):
        ds_data = gve[gve["dataset"] == ds].sort_values("n_threads")
        if len(ds_data) < 2:
            continue
        ax.plot(
            ds_data["n_threads"],
            ds_data["mean_time"],
            marker=markers[i % len(markers)],
            linewidth=2,
            label=ds,
        )

    ax.set_xlabel("Number of Threads", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("GVE-Leiden: Thread Scaling", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "benchmark_thread_scaling.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_modularity(df, resolution=1.0):
    """Compare modularity across implementations."""
    sub = df[(df["resolution"] == resolution) & (df["modularity"].notna())].copy()
    if sub.empty:
        sub = df[df["modularity"].notna()].copy()
    if sub.empty:
        print("  No modularity data available.")
        return

    # For gve-leiden, pick best thread count
    gve = sub[sub["tool"] == "gve-leiden"]
    if not gve.empty:
        for ds in gve["dataset"].unique():
            ds_gve = gve[gve["dataset"] == ds]
            best_idx = ds_gve["mean_time"].idxmin()
            drop_idx = ds_gve.index[ds_gve.index != best_idx]
            sub = sub.drop(drop_idx)

    fig, ax = plt.subplots(figsize=(10, 5))

    datasets_present = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    impls_present = [i for i in IMPL_ORDER if i in sub["impl_label"].unique()]

    x = np.arange(len(datasets_present))
    width = 0.15
    n_impls = len(impls_present)
    offsets = np.arange(n_impls) - (n_impls - 1) / 2

    for i, impl in enumerate(impls_present):
        mods = []
        for ds in datasets_present:
            row = sub[(sub["dataset"] == ds) & (sub["impl_label"] == impl)]
            mods.append(row.iloc[0]["modularity"] if not row.empty else 0)
        color = IMPL_COLORS.get(impl, "#999999")
        ax.bar(x + offsets[i] * width, mods, width, label=impl, color=color, edgecolor="white")

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Modularity", fontsize=12)
    ax.set_title(f"Clustering Quality: Modularity (resolution={resolution})", fontsize=14)
    ax.set_xticks(x)
    dataset_labels = get_dataset_labels_with_counts(df)
    ax.set_xticklabels([dataset_labels.get(d, d) for d in datasets_present])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "benchmark_modularity.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_peak_memory(df):
    """Plot peak memory usage across implementations and datasets."""
    sub = df.copy()
    
    # For gve-leiden, pick the run with most threads
    gve = sub[sub["tool"] == "gve-leiden"]
    if not gve.empty:
        for ds in gve["dataset"].unique():
            ds_gve = gve[gve["dataset"] == ds]
            best_idx = ds_gve["mean_time"].idxmin()
            drop_idx = ds_gve.index[ds_gve.index != best_idx]
            sub = sub.drop(drop_idx)
    
    # Remove rows with missing peak_memory_mb
    sub = sub[sub["peak_memory_mb"].notna()].copy()
    
    if sub.empty:
        print("  No peak memory data available.")
        return
    
    datasets_present = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    impls_present = [i for i in IMPL_ORDER if i in sub["impl_label"].unique()]
    
    if not datasets_present or not impls_present:
        print("  Not enough data to plot peak memory.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(datasets_present))
    width = 0.15
    n_impls = len(impls_present)
    offsets = np.arange(n_impls) - (n_impls - 1) / 2
    
    for i, impl in enumerate(impls_present):
        mems = []
        for ds in datasets_present:
            row = sub[(sub["dataset"] == ds) & (sub["impl_label"] == impl)]
            if not row.empty:
                mems.append(row.iloc[0]["peak_memory_mb"])
            else:
                mems.append(0)
        color = IMPL_COLORS.get(impl, "#999999")
        ax.bar(x + offsets[i] * width, mems, width, label=impl, color=color, edgecolor="white")
    
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax.set_title("Peak Memory Usage by Implementation and Dataset", fontsize=14)
    ax.set_xticks(x)
    dataset_labels = get_dataset_labels_with_counts(df)
    ax.set_xticklabels([dataset_labels.get(d, d) for d in datasets_present])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "benchmark_peak_memory.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_resolution_vs_clusters(df):
    """Plot clustering count vs. resolution for Tabula Sapiens across implementations."""
    # Filter to Tabula Sapiens only
    sub = df[df["dataset"] == "tabula_sapiens"].copy()
    
    if sub.empty:
        print("  No Tabula Sapiens data available for resolution vs. clusters plot.")
        return
    
    # For gve-leiden, pick the run with most threads
    gve = sub[sub["tool"] == "gve-leiden"]
    if not gve.empty:
        for res in gve["resolution"].unique():
            res_gve = gve[gve["resolution"] == res]
            best_idx = res_gve["mean_time"].idxmin()
            drop_idx = res_gve.index[res_gve.index != best_idx]
            sub = sub.drop(drop_idx)
    
    # Get unique resolutions and implementations
    resolutions = sorted(sub["resolution"].unique())
    impls_present = [i for i in IMPL_ORDER if i in sub["impl_label"].unique()]
    
    if not resolutions or not impls_present:
        print("  Not enough data to plot resolution vs. clusters.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for impl in impls_present:
        impl_data = sub[sub["impl_label"] == impl]
        clusters = []
        for res in resolutions:
            row = impl_data[impl_data["resolution"] == res]
            if not row.empty:
                clusters.append(row.iloc[0]["n_clusters"])
            else:
                clusters.append(0)
        
        color = IMPL_COLORS.get(impl, "#999999")
        ax.plot(resolutions, clusters, marker="o", label=impl, color=color, linewidth=2, markersize=8)
    
    ax.set_xlabel("Resolution Parameter", fontsize=12)
    ax.set_ylabel("Number of Clusters", fontsize=12)
    ax.set_title("Clustering Resolution vs. Number of Clusters (Tabula Sapiens)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "benchmark_resolution_vs_clusters.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("=" * 60)
    print("Benchmark Results Aggregation & Visualization")
    print("=" * 60)

    df = load_results()
    if df.empty:
        print("\nNo results found. Run the benchmark scripts first.")
        sys.exit(1)

    print(f"\nLoaded {len(df)} result entries.")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Implementations: {sorted(df['impl_label'].unique())}")

    csv_path = os.path.join(RESULTS_DIR, "benchmark_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary CSV: {csv_path}")

    # Print summary table
    print("\n--- Summary Table ---")
    summary_cols = ["dataset", "impl_label", "n_threads", "resolution",
                    "mean_time", "std_time", "n_clusters", "modularity"]
    available_cols = [c for c in summary_cols if c in df.columns]
    print(df[available_cols].to_string(index=False))

    # Print per-run timing table
    print("\n--- Per-Run Timing (seconds) ---")
    for _, row in df.iterrows():
        times = row.get("individual_times", [])
        if not times:
            times_str = "N/A"
        else:
            times_str = ", ".join(f"{t:.4f}" for t in times)
        print(f"  {row['dataset']:20s}  {row['impl_label']:25s}  "
              f"res={row['resolution']:<5}  "
              f"runs: [{times_str}]  "
              f"mean={row['mean_time']:.4f}s")

    # Generate plots
    print("\nGenerating plots...")
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)

    plot_runtime_comparison(df, resolution=1.0)
    plot_scalability(df, resolution=1.0)
    plot_thread_scaling(df)
    plot_modularity(df, resolution=1.0)
    plot_peak_memory(df)
    plot_resolution_vs_clusters(df)

    print("\nDone! Check results/ for outputs.")


if __name__ == "__main__":
    main()
