#!/usr/bin/env python3
"""
Benchmark Scanpy Leiden clustering implementations.

Tests two flavors:
  - leidenalg (Python leidenalg package via scanpy)
  - igraph    (C igraph core via scanpy)

For each dataset, the script:
  1. Loads / preprocesses the data (normalize, HVG, PCA, neighbors)
  2. Saves the preprocessed AnnData so other benchmarks can reuse it
  3. Runs Leiden clustering with each flavor, measuring time and metrics
  4. Exports the SNN graph in MatrixMarket format for GVE-Leiden
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import psutil
import scanpy as sc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

N_TOP_GENES = 2000
N_PCS = 50
N_NEIGHBORS = 15
RESOLUTIONS = [0.5, 1.0, 2.0]
N_REPEATS = 3


def get_memory_mb():
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def load_pbmc3k():
    """Load PBMC 3k from 10x filtered matrices."""
    mtx_dir = os.path.join(DATA_DIR, "filtered_gene_bc_matrices", "hg19")
    if not os.path.isdir(mtx_dir):
        alt = os.path.join(DATA_DIR, "pbmc3k_filtered_gene_bc_matrices.tar.gz")
        if os.path.exists(alt):
            import tarfile
            with tarfile.open(alt, "r:gz") as tar:
                tar.extractall(path=DATA_DIR)
    if os.path.isdir(mtx_dir):
        adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=True)
    else:
        print("  PBMC 3k raw data not found, using scanpy built-in pbmc3k_processed")
        adata = sc.datasets.pbmc3k_processed()
        return adata
    adata.var_names_make_unique()
    return adata


def load_pbmc10k():
    """Load PBMC 10k from 10x h5 or h5ad file."""
    # Check for h5ad first (user-provided or preprocessed)
    h5ad_path = os.path.join(DATA_DIR, "pbmc_10k_v3_filtered_feature_bc_matrix.h5ad")
    if os.path.exists(h5ad_path):
        adata = sc.read_h5ad(h5ad_path)
        adata.var_names_make_unique()
        return adata

    # Fall back to 10x h5 format
    h5_path = os.path.join(DATA_DIR, "pbmc_10k_v3_filtered_feature_bc_matrix.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"PBMC 10k not found. Expected either:\n"
            f"  {h5ad_path}\n"
            f"  {h5_path}\n"
            f"Run download_data.py or place your h5ad file in data/ folder."
        )
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    return adata


def load_tabula_sapiens():
    """Load Tabula Sapiens h5ad (preprocessed from figshare download)."""
    for candidate in [
        os.path.join(DATA_DIR, "tabula_sapiens_preprocessed.h5ad"),
        os.path.join(DATA_DIR, "TabulaSapiens.h5ad"),
    ]:
        if os.path.exists(candidate):
            adata = sc.read_h5ad(candidate)
            return adata

    h5_files = [
        f for f in os.listdir(DATA_DIR)
        if "tabula" in f.lower() and f.endswith((".h5ad", ".h5"))
    ]
    if h5_files:
        path = os.path.join(DATA_DIR, sorted(h5_files)[0])
        print(f"  Loading Tabula Sapiens from {path}")
        if path.endswith(".h5ad"):
            return sc.read_h5ad(path)
        else:
            return sc.read_10x_h5(path)

    raise FileNotFoundError(
        "Tabula Sapiens data not found. Download from figshare or CZ CELLxGENE."
    )


DATASET_LOADERS = {
    "pbmc3k": load_pbmc3k,
    "pbmc10k": load_pbmc10k,
    "tabula_sapiens": load_tabula_sapiens,
}


def preprocess(adata, dataset_name):
    """Standard scRNA-seq preprocessing: filter, normalize, HVG, PCA, neighbors."""
    preprocessed_path = os.path.join(DATA_DIR, f"{dataset_name}_preprocessed.h5ad")
    if os.path.exists(preprocessed_path):
        print(f"  Loading preprocessed data from {preprocessed_path}")
        return sc.read_h5ad(preprocessed_path)

    # If the data already has a neighbor graph (e.g. scanpy's pbmc3k_processed),
    # it's already preprocessed — just save and return it.
    if "neighbors" in adata.uns and "connectivities" in adata.obsp:
        print(f"  Data already preprocessed ({adata.n_obs} cells). Skipping pipeline.")
        adata.write(preprocessed_path)
        return adata

    print(f"  Preprocessing {dataset_name} ({adata.n_obs} cells x {adata.n_vars} genes)")

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=False)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(N_TOP_GENES, adata.n_vars))
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=min(N_PCS, adata.n_vars - 1))
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=min(N_PCS, adata.n_vars - 1))

    print(f"  After preprocessing: {adata.n_obs} cells x {adata.n_vars} genes")
    adata.write(preprocessed_path)
    print(f"  Saved preprocessed data to {preprocessed_path}")
    return adata


def export_snn_graph(adata, dataset_name):
    """Export the SNN connectivities graph as MatrixMarket for GVE-Leiden."""
    from scipy.io import mmwrite
    from scipy.sparse import triu

    mtx_path = os.path.join(DATA_DIR, f"{dataset_name}_snn.mtx")
    if os.path.exists(mtx_path):
        print(f"  SNN graph already exported: {mtx_path}")
        return mtx_path

    adj = adata.obsp["connectivities"]
    sym = (adj + adj.T) / 2
    upper = triu(sym)

    mmwrite(mtx_path, upper, comment=f"SNN graph for {dataset_name}")
    print(f"  Exported SNN graph ({adj.shape[0]} nodes) to {mtx_path}")
    return mtx_path


def _leiden_leidenalg(adata, resolution, random_state):
    """Call leidenalg directly (works on all scanpy versions)."""
    import leidenalg
    import igraph as ig

    adjacency = adata.obsp["connectivities"]
    sources, targets = adjacency.nonzero()
    weights = np.array(adjacency[sources, targets]).flatten()

    g = ig.Graph(directed=False)
    g.add_vertices(adjacency.shape[0])
    g.add_edges(list(zip(sources.tolist(), targets.tolist())))
    g.es["weight"] = weights.tolist()

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=2,
        seed=random_state,
    )
    labels = [str(m) for m in partition.membership]
    adata.obs["leiden_bench"] = labels
    adata.obs["leiden_bench"] = adata.obs["leiden_bench"].astype("category")
    return partition.modularity


def _leiden_igraph(adata, resolution, random_state):
    """Call igraph community_leiden directly (works on all scanpy versions)."""
    import igraph as ig
    import random as rnd

    adjacency = adata.obsp["connectivities"]
    sources, targets = adjacency.nonzero()
    weights = np.array(adjacency[sources, targets]).flatten()

    g = ig.Graph(directed=False)
    g.add_vertices(adjacency.shape[0])
    g.add_edges(list(zip(sources.tolist(), targets.tolist())))
    g.es["weight"] = weights.tolist()

    rnd.seed(random_state)
    cl = g.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=resolution,
        n_iterations=2,
    )
    labels = [str(m) for m in cl.membership]
    adata.obs["leiden_bench"] = labels
    adata.obs["leiden_bench"] = adata.obs["leiden_bench"].astype("category")
    return cl.modularity


_FLAVOR_FN = {
    "leidenalg": _leiden_leidenalg,
    "igraph": _leiden_igraph,
}


def benchmark_leiden(adata, flavor, resolution, n_repeats=N_REPEATS):
    """Run Leiden clustering and collect timing + quality metrics."""
    times = []
    n_clusters_list = []
    modularity_list = []

    leiden_fn = _FLAVOR_FN[flavor]

    for i in range(n_repeats):
        gc.collect()
        mem_before = get_memory_mb()

        t0 = time.perf_counter()
        mod = leiden_fn(adata, resolution, random_state=42 + i)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        mem_after = get_memory_mb()

        n_clust = adata.obs["leiden_bench"].nunique()
        if mod is not None:
            modularity_list.append(float(mod))

        times.append(elapsed)
        n_clusters_list.append(n_clust)

        print(f"      Run {i+1}/{n_repeats}: {elapsed:.4f}s "
              f"(clusters={n_clust}, mem={mem_after:.0f}MB)")

    return {
        "times": times,
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "min_time": float(np.min(times)),
        "n_clusters": int(n_clusters_list[-1]),
        "modularity": float(np.mean(modularity_list)) if modularity_list else None,
        "peak_memory_mb": mem_after,
    }


def run_benchmarks(datasets=None, resolutions=None, preprocess_only=False, n_repeats=N_REPEATS):
    """Main benchmark runner."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    datasets = datasets or list(DATASET_LOADERS.keys())
    resolutions = resolutions or RESOLUTIONS

    all_results = []

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            adata_raw = DATASET_LOADERS[ds_name]()
        except (FileNotFoundError, Exception) as e:
            print(f"  [SKIP] Could not load {ds_name}: {e}")
            continue

        adata = preprocess(adata_raw, ds_name)
        export_snn_graph(adata, ds_name)

        if preprocess_only:
            print(f"  Preprocessing complete for {ds_name}. Skipping benchmarks.")
            continue

        n_cells = adata.n_obs
        print(f"  Benchmarking on {n_cells} cells...")

        for flavor in ["leidenalg", "igraph"]:
            for res in resolutions:
                label = f"scanpy_{flavor}_res{res}"
                print(f"\n  [{label}] resolution={res}, n_repeats={n_repeats}")

                try:
                    metrics = benchmark_leiden(adata, flavor, res, n_repeats=n_repeats)
                except Exception as e:
                    print(f"    [ERROR] {e}")
                    continue

                result = {
                    "tool": "scanpy",
                    "implementation": flavor,
                    "dataset": ds_name,
                    "n_cells": n_cells,
                    "resolution": res,
                    "n_repeats": n_repeats,
                    **metrics,
                }
                all_results.append(result)

                print(f"    Time: {metrics['mean_time']:.3f}s "
                      f"(±{metrics['std_time']:.3f}s)")
                print(f"    Individual runs: "
                      + ", ".join(f"{t:.4f}s" for t in metrics['times']))
                print(f"    Clusters: {metrics['n_clusters']}")
                if metrics["modularity"] is not None:
                    print(f"    Modularity: {metrics['modularity']:.4f}")

                result_file = os.path.join(
                    RESULTS_DIR, f"{ds_name}_{label}.json"
                )
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)

    summary_path = os.path.join(RESULTS_DIR, "scanpy_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll Scanpy results saved to {summary_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Scanpy Leiden clustering")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        choices=list(DATASET_LOADERS.keys()),
        help="Datasets to benchmark (default: all available)",
    )
    parser.add_argument(
        "--resolutions", nargs="+", type=float, default=RESOLUTIONS,
        help=f"Resolution values (default: {RESOLUTIONS})",
    )
    parser.add_argument(
        "--preprocess-only", action="store_true",
        help="Only preprocess data; skip benchmarks",
    )
    parser.add_argument(
        "--repeats", type=int, default=N_REPEATS,
        help=f"Number of repetitions per configuration (default: {N_REPEATS})",
    )
    args = parser.parse_args()

    sc.settings.verbosity = 1
    run_benchmarks(
        datasets=args.datasets,
        resolutions=args.resolutions,
        preprocess_only=args.preprocess_only,
        n_repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
