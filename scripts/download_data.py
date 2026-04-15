#!/usr/bin/env python3
"""
Download publicly available scRNA-seq datasets for benchmarking.

Datasets:
  1. PBMC 3k  (~2,700 cells)   — 10x Genomics (raw MTX)
  2. PBMC 10k (~11,769 cells)  — 10x Genomics v3 chemistry (h5)
  3. Tabula Sapiens (~483k cells) — via CZ CELLxGENE Census API

Usage:
  python scripts/download_data.py                     # all datasets
  python scripts/download_data.py pbmc3k pbmc10k      # specific datasets
  python scripts/download_data.py tabula_sapiens       # large dataset only

After this script completes, run:
  python scripts/benchmark_scanpy.py --preprocess-only
"""

import os
import sys
import tarfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


# ---------------------------------------------------------------------------
# PBMC 3k
# ---------------------------------------------------------------------------

def download_pbmc3k():
    dest = os.path.join(DATA_DIR, "pbmc3k_filtered_gene_bc_matrices.tar.gz")
    mtx_dir = os.path.join(DATA_DIR, "filtered_gene_bc_matrices", "hg19")

    if os.path.isdir(mtx_dir):
        print("  [skip] PBMC 3k already extracted.")
        return True

    url = "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
    if not os.path.exists(dest):
        ok = _download(url, dest, "10x PBMC 3k raw matrices (~6 MB)")
        if not ok:
            return False

    print("  Extracting pbmc3k_filtered_gene_bc_matrices.tar.gz ...")
    with tarfile.open(dest, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    print("  Done.")
    return True


# ---------------------------------------------------------------------------
# PBMC 10k
# ---------------------------------------------------------------------------

def download_pbmc10k():
    dest = os.path.join(DATA_DIR, "pbmc_10k_v3_filtered_feature_bc_matrix.h5")
    if os.path.exists(dest):
        print("  [skip] PBMC 10k already downloaded.")
        return True

    url = ("https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/"
           "pbmc_10k_v3_filtered_feature_bc_matrix.h5")
    return _download(url, dest, "10x PBMC 10k v3 h5 (~55 MB)")


# ---------------------------------------------------------------------------
# Tabula Sapiens  — downloaded via CZ CELLxGENE Census API
# ---------------------------------------------------------------------------

def download_tabula_sapiens():
    """
    Fetch a memory-safe subset of Tabula Sapiens from CZ CELLxGENE Census.

    The full atlas (~483k cells × 61k genes) requires >100 GB RAM and causes
    an OOM kill on typical workstations.  This function avoids that by:

      1. Discovering Tabula Sapiens dataset IDs dynamically from census metadata
      2. Computing HVGs SERVER-SIDE via census experimental API (no local matrix)
      3. Fetching only HVG-filtered genes + a random subsample of cells
         → final matrix is ~100k × 2k, fits comfortably in 2–4 GB RAM
      4. Running normalize → PCA → neighbors locally, then saving h5ad

    Requires:  pip install cellxgene-census
    """
    MAX_CELLS   = 100_000
    N_TOP_GENES = 2_000

    out_path = os.path.join(DATA_DIR, "tabula_sapiens_preprocessed.h5ad")
    if os.path.exists(out_path):
        print(f"  [skip] Already exists: {out_path}")
        return True

    try:
        import cellxgene_census
        from cellxgene_census.experimental.pp import highly_variable_genes as census_hvg
    except ImportError:
        print("  [ERROR] cellxgene-census not installed.")
        print("  Install with:  pip install cellxgene-census")
        print("  Then re-run:   python scripts/download_data.py tabula_sapiens")
        return False

    import random
    import scanpy as sc

    print("  Opening CZ CELLxGENE Census (stable release)...")

    with cellxgene_census.open_soma(census_version="stable") as census:

        # ── 1. Find Tabula Sapiens dataset IDs dynamically ────────────────────
        print("  Looking up Tabula Sapiens dataset IDs...")
        datasets_df = (
            census["census_info"]["datasets"]
            .read()
            .concat()
            .to_pandas()
        )
        ts_rows = datasets_df[
            datasets_df["collection_name"].str.contains(
                "Tabula Sapiens", case=False, na=False
            )
        ]
        if ts_rows.empty:
            ts_rows = datasets_df[
                datasets_df["dataset_title"].str.contains(
                    "Tabula Sapiens", case=False, na=False
                )
            ]
        if ts_rows.empty:
            raise RuntimeError(
                "Could not find Tabula Sapiens in the current Census release.\n"
                "Available collections:\n"
                + datasets_df["collection_name"].drop_duplicates().to_string()
            )

        ts_ids = ts_rows["dataset_id"].tolist()
        print(f"  Found {len(ts_ids)} Tabula Sapiens dataset(s): {ts_ids}")
        id_list = ", ".join(f'"{d}"' for d in ts_ids)
        obs_filter = f"dataset_id in [{id_list}]"

        # ── 2. Compute HVGs server-side — avoids loading the full gene matrix ─
        print(f"  Computing top {N_TOP_GENES} HVGs server-side (may take ~5 min)...")
        import tiledbsoma as soma

        experiment = census["census_data"]["homo_sapiens"]
        with experiment.axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=obs_filter),
        ) as query:
            hvg_df = census_hvg(query, n_top_genes=N_TOP_GENES)

        # hvg_df is indexed by soma_joinid (integer) -- pass directly as
        # var_coords to get_anndata, no filter-string building needed.
        hv_var_joinids = hvg_df.index[hvg_df["highly_variable"]].tolist()
        print(f"  Selected {len(hv_var_joinids)} highly variable genes")

        # ── 3. Subsample cells to stay within RAM budget ──────────────────────
        print("  Reading cell IDs for subsampling...")
        all_joinids = (
            census["census_data"]["homo_sapiens"]
            .obs.read(
                value_filter=obs_filter,
                column_names=["soma_joinid"],
            )
            .concat()
            .to_pandas()["soma_joinid"]
            .tolist()
        )
        n_total = len(all_joinids)
        print(f"  Total Tabula Sapiens cells in census: {n_total:,}")
        if n_total > MAX_CELLS:
            random.seed(42)
            all_joinids = random.sample(all_joinids, MAX_CELLS)
            print(f"  Subsampled to {MAX_CELLS:,} cells")

        # ── 4. Fetch cells × HVGs only (~100–500 MB download) ────────────────
        print(f"  Fetching {len(all_joinids):,} cells x {len(hv_var_joinids)} HVGs...")
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_coords=all_joinids,
            var_coords=hv_var_joinids,
            obs_column_names=[
                "cell_type", "tissue", "tissue_general",
                "disease", "sex", "donor_id",
            ],
        )

    print(f"  Downloaded: {adata.n_obs:,} cells × {adata.n_vars} genes")
    if adata.n_obs == 0:
        raise RuntimeError("Downloaded 0 cells — subsampling or filter issue.")

    # ── 5. Preprocess: HVGs already selected, skip re-running HVG detection ──
    print("  Preprocessing (normalize → scale → PCA → neighbors)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

    print(f"  Saving to {out_path} ...")
    adata.write_h5ad(out_path)
    print(f"  Saved ({os.path.getsize(out_path) / 1e9:.2f} GB)")
    return True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DOWNLOADERS = {
    "pbmc3k":         download_pbmc3k,
    "pbmc10k":        download_pbmc10k,
    "tabula_sapiens": download_tabula_sapiens,
}

DESCRIPTIONS = {
    "pbmc3k":         "10x PBMC 3k   (~2,700 cells,  ~6 MB)",
    "pbmc10k":        "10x PBMC 10k  (~11,769 cells, ~55 MB)",
    "tabula_sapiens": "Tabula Sapiens (100k cells subsampled, ~2k HVGs via Census API)",
}


def _download(url, dest, description=""):
    if os.path.exists(dest):
        print(f"  [skip] Already exists: {dest}")
        return True

    print(f"  Downloading {description} ...")
    print(f"    → {dest}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        print()
        return True
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    mb = downloaded / (1024 * 1024)
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r    {pct:5.1f}%  ({mb:.1f} / {total_mb:.1f} MB)")
    else:
        sys.stdout.write(f"\r    {mb:.1f} MB downloaded")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    requested = sys.argv[1:] if len(sys.argv) > 1 else list(DOWNLOADERS.keys())

    unknown = [n for n in requested if n not in DOWNLOADERS]
    if unknown:
        print(f"[ERROR] Unknown datasets: {unknown}")
        print(f"  Valid options: {list(DOWNLOADERS.keys())}")
        sys.exit(1)

    print("=" * 60)
    print("scRNA-seq Benchmark Data Downloader")
    print("=" * 60)
    for name in requested:
        print(f"  • {name:20s} {DESCRIPTIONS[name]}")
    print()

    results = {}
    for name in requested:
        print(f"\n{'─'*60}")
        print(f"Dataset: {name}  —  {DESCRIPTIONS[name]}")
        print(f"{'─'*60}")
        results[name] = DOWNLOADERS[name]()

    print("\n" + "=" * 60)
    print("Summary:")
    for name, ok in results.items():
        status = "✓ ready" if ok else "✗ failed"
        print(f"  {name:20s}  {status}")

    print("\nNext step — preprocess all downloaded datasets:")
    print("  python scripts/benchmark_scanpy.py --preprocess-only")
    print("\nOr run the full benchmark suite:")
    print("  make all")


if __name__ == "__main__":
    main()
