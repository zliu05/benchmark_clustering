.PHONY: all download preprocess benchmark-scanpy benchmark-seurat benchmark-gveleiden \
        plots clean setup-conda

# Override with: make PYTHON=/path/to/python
# If using the benchmark_leiden conda env: make PYTHON=$(conda run -n benchmark_leiden which python)
PYTHON ?= python3
RSCRIPT ?= Rscript

# ---- Setup ----

# Install gcc via conda so GVE-Leiden can compile with OpenMP
setup-conda:
	conda install -y -c conda-forge llvm-openmp
	@echo "llvm-openmp installed. Now run: make benchmark-gveleiden"

# ---- Top-level targets ----

all: download benchmark-scanpy benchmark-seurat benchmark-gveleiden plots

# ---- Data ----

download:
	$(PYTHON) scripts/download_data.py pbmc3k pbmc10k
	@echo "---"
	@echo "Note: Tabula Sapiens (~483k cells) is a large download."
	@echo "To include it, run: $(PYTHON) scripts/download_data.py tabula_sapiens"

download-all:
	$(PYTHON) scripts/download_data.py

preprocess:
	$(PYTHON) scripts/benchmark_scanpy.py --preprocess-only

# ---- Benchmarks ----

benchmark-scanpy: preprocess
	$(PYTHON) scripts/benchmark_scanpy.py

benchmark-seurat: preprocess
	$(RSCRIPT) scripts/benchmark_seurat.R

benchmark-gveleiden: preprocess
	$(PYTHON) scripts/benchmark_gveleiden.py

# Run only on small datasets (fast, for testing)
benchmark-quick:
	$(PYTHON) scripts/benchmark_scanpy.py --datasets pbmc3k --resolutions 1.0 --repeats 1
	$(RSCRIPT) scripts/benchmark_seurat.R pbmc3k
	$(PYTHON) scripts/benchmark_gveleiden.py --datasets pbmc3k --repeats 1

# ---- Plots ----

plots:
	$(PYTHON) scripts/plot_results.py

# ---- Cleanup ----

clean:
	rm -rf results/*.json results/*.csv results/*.png

clean-data:
	rm -rf data/

clean-all: clean clean-data
	rm -rf gve-leiden/
