# Leiden Clustering Benchmark Pipeline

Benchmarking pipeline comparing Leiden community detection implementations
across different frameworks on scRNA-seq data at multiple scales.

## Implementations Compared

| Implementation | Language | Backend | Parallelism |
|---|---|---|---|
| Scanpy (leidenalg) | Python | leidenalg package | Single-threaded |
| Scanpy (igraph) | Python | igraph C core | Single-threaded |
| Seurat (leidenbase) | R | leidenbase (C/C++) | Single-threaded |
| Seurat (igraph) | R | igraph R/C | Single-threaded |
| GVE-Leiden | C++ | OpenMP parallel | Multi-threaded |

## Benchmark Datasets

We use publicly available scRNA-seq datasets at three scales:

| Dataset | Cells | Source | Reference |
|---|---|---|---|
| PBMC 3k | 2,638 | 10x Genomics | [Link](https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz) |
| PBMC 10k | 374,702 | 10x Genomics (v3 chemistry) | [Link](https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5) |
| Tabula Sapiens | 100,000 | CZ CELLxGENE / Figshare | [Tabula Sapiens Consortium, Science 2022](https://doi.org/10.1126/science.abl4896) |

## Metrics Collected

- **Wall-clock time** (seconds): Total time for Leiden clustering only
- **Number of clusters**: Communities detected
- **Modularity**: Quality of the partition
- **Peak memory** (MB): Maximum RSS during clustering

## Prerequisites

### Python
```bash
pip install -r requirements.txt
```

### R
```r
install.packages(c("Seurat", "SeuratObject", "leidenbase", "igraph",
                    "Matrix", "jsonlite", "tictoc", "pryr"))
```

### GVE-Leiden (C++)
Requires a C++17 compiler with OpenMP support. The benchmark script
auto-detects in this order: conda gcc → Homebrew gcc → Apple Clang + libomp.

**Recommended — conda (macOS Apple Silicon & Intel)**
```bash
conda install -c conda-forge llvm-openmp
```
This enables OpenMP on Apple Clang — no gcc cross-compiler needed.
`llvm-openmp` installs `omp.h` and `libomp.dylib` into the conda env prefix,
and the benchmark script picks them up automatically.

**Linux via conda**
```bash
conda install -c conda-forge gxx
```

**macOS via Homebrew (alternative)**
```bash
brew install libomp     # Apple Clang + libomp
brew install gcc        # or: full GNU gcc with OpenMP
```

## Usage

### Quick Start
```bash
# 1. Download datasets
python scripts/download_data.py

# 2. Run all benchmarks
make all

# 3. Or run individual benchmarks
make benchmark-scanpy
make benchmark-seurat
make benchmark-gveleiden

# 4. Generate plots
make plots
```

### Run Step-by-Step
```bash
# Download data
python scripts/download_data.py

# Python benchmarks (Scanpy leidenalg and igraph)
python scripts/benchmark_scanpy.py

# R benchmarks (Seurat leidenbase and igraph)
Rscript scripts/benchmark_seurat.R

# GVE-Leiden benchmark
# (automatically compiles, converts graph to MTX, and runs)
python scripts/benchmark_gveleiden.py

# Aggregate results and plot
python scripts/plot_results.py
```

## Output

Results are saved to `results/` as JSON files per run, then aggregated into:
- `results/benchmark_summary.csv` — tabular summary
- `results/benchmark_runtime.png` — runtime comparison bar chart
- `results/benchmark_scalability.png` — scalability plot (cells vs time)

## Project Structure

```
├── README.md
├── Makefile
├── requirements.txt
├── scripts/
│   ├── download_data.py         # Dataset downloader
│   ├── benchmark_scanpy.py      # Scanpy benchmarks (leidenalg + igraph)
│   ├── benchmark_seurat.R       # Seurat benchmarks (leidenbase + igraph)
│   ├── benchmark_gveleiden.py   # GVE-Leiden wrapper
│   ├── export_graph_mtx.py      # Export SNN graph to MatrixMarket for GVE-Leiden
│   └── plot_results.py          # Aggregate & visualize
├── data/                        # Downloaded datasets (gitignored)
├── results/                     # Benchmark outputs (gitignored)
└── gve-leiden/                  # Cloned GVE-Leiden repo (gitignored)
```

## References

- Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden: guaranteeing well-connected communities. *Sci Rep* 9, 5233 (2019).
- GVE-Leiden: [arxiv.org/abs/2312.13936](https://arxiv.org/abs/2312.13936)
- The Tabula Sapiens Consortium. *Science* 376, eabl4896 (2022).
- Wolf, F.A., Angerer, P. & Theis, F.J. SCANPY: large-scale single-cell gene expression data analysis. *Genome Biol* 19, 15 (2018).
- Hao, Y. et al. Dictionary learning for integrative, multimodal and scalable single-cell analysis. *Nat Biotechnol* 42, 293–304 (2024).
