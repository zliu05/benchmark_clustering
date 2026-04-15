 #!/usr/bin/env Rscript
#
# Benchmark Seurat Leiden clustering implementations.
#
# Tests two backends:
#   - leidenbase (C/C++ via leidenbase package, default in Seurat v5)
#   - igraph     (R igraph C core, via cluster_leiden)
#
# For each preprocessed dataset (h5ad exported from the Python pipeline),
# loads the SNN graph and runs Leiden with each method.
#
# Dependencies:
#   install.packages(c("Seurat", "SeuratObject", "SeuratDisk", "leidenbase",
#                       "igraph", "Matrix", "jsonlite", "tictoc", "pryr", "anndata"))
#   -- or use reticulate to read h5ad via anndata

suppressPackageStartupMessages({
    library(Seurat)
    library(Matrix)
    library(igraph)
    library(jsonlite)
})

# -- Configuration -----------------------------------------------------------
# Robustly find script location whether run via Rscript or source()
.get_script_dir <- function() {
    # When run via Rscript --vanilla scripts/foo.R
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args, value = TRUE)
    if (length(file_arg) > 0) {
        return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
    }
    # When run via source() inside an R session
    src <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) NULL)
    if (!is.null(src)) return(src)
    # Fallback: current working directory
    getwd()
}
SCRIPT_DIR  <- .get_script_dir()
PROJECT_DIR <- normalizePath(file.path(SCRIPT_DIR, ".."))
DATA_DIR    <- file.path(PROJECT_DIR, "data")
RESULTS_DIR <- file.path(PROJECT_DIR, "results")

RESOLUTIONS <- c(0.5, 1.0, 2.0)
N_REPEATS   <- 3
RANDOM_SEED <- 42

DATASETS <- c("pbmc3k", "pbmc10k", "tabula_sapiens")

# -- Helpers -----------------------------------------------------------------
`%||%` <- function(a, b) if (is.null(a)) b else a

get_memory_mb <- function() {
    gc(verbose = FALSE)
    as.numeric(system(
        paste("ps -o rss= -p", Sys.getpid()),
        intern = TRUE
    )) / 1024
}

save_result <- function(result, filename) {
    dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
    path <- file.path(RESULTS_DIR, filename)
    write(toJSON(result, auto_unbox = TRUE, pretty = TRUE), path)
    cat("    Saved:", path, "\n")
}

# -- Load preprocessed data via SNN graph from MTX ---------------------------
load_snn_from_mtx <- function(dataset_name) {
    mtx_path <- file.path(DATA_DIR, paste0(dataset_name, "_snn.mtx"))
    if (!file.exists(mtx_path)) {
        stop(paste("SNN graph not found:", mtx_path,
                    "\nRun benchmark_scanpy.py first to preprocess and export graphs."))
    }

    cat("  Loading SNN graph from", mtx_path, "\n")
    adj <- readMM(mtx_path)

    # readMM gives upper triangular; symmetricize
    adj <- adj + t(adj)
    diag(adj) <- 0

    n <- nrow(adj)
    cat("  Graph:", n, "nodes,", nnzero(adj), "non-zero edges\n")
    return(adj)
}

# -- Load full Seurat object for richer benchmarking -------------------------
load_seurat_from_h5ad <- function(dataset_name) {
    h5ad_path <- file.path(DATA_DIR, paste0(dataset_name, "_preprocessed.h5ad"))

    if (!file.exists(h5ad_path)) {
        stop(paste("Preprocessed h5ad not found:", h5ad_path,
                    "\nRun benchmark_scanpy.py first."))
    }

    # Use anndata via reticulate to load, then convert to Seurat
    tryCatch({
        ad <- reticulate::import("anndata")
        adata <- ad$read_h5ad(h5ad_path)

        # Extract the connectivities as the SNN graph
        conn <- adata$obsp["connectivities"]
        # Convert scipy sparse to R dgCMatrix
        scipy_sparse <- reticulate::import("scipy.sparse")
        csc <- scipy_sparse$csc_matrix(conn)
        adj <- as(
            sparseMatrix(
                i = as.integer(csc$tocoo()$row) + 1L,
                j = as.integer(csc$tocoo()$col) + 1L,
                x = as.numeric(csc$tocoo()$data),
                dims = as.integer(csc$shape)
            ),
            "dgCMatrix"
        )
        cell_names <- adata$obs_names$tolist()
        rownames(adj) <- cell_names
        colnames(adj) <- cell_names

        return(list(adj = adj, n_cells = length(cell_names)))
    }, error = function(e) {
        cat("  Could not load via reticulate/anndata:", conditionMessage(e), "\n")
        cat("  Falling back to MTX graph only.\n")
        adj <- load_snn_from_mtx(dataset_name)
        return(list(adj = adj, n_cells = nrow(adj)))
    })
}

# -- Benchmark with leidenbase -----------------------------------------------
benchmark_leidenbase <- function(adj, resolution, n_repeats = N_REPEATS) {
    if (!requireNamespace("leidenbase", quietly = TRUE)) {
        stop("leidenbase package not installed. Install via:\n",
             "  install.packages('leidenbase')")
    }

    g <- graph_from_adjacency_matrix(adj, mode = "undirected", weighted = TRUE)

    times <- numeric(n_repeats)
    n_clusters <- integer(n_repeats)

    for (i in seq_len(n_repeats)) {
        gc(verbose = FALSE)
        t0 <- proc.time()

        partition <- leidenbase::leiden_find_partition(
            g,
            partition_type = "RBConfigurationVertexPartition",
            resolution_parameter = resolution,
            seed = RANDOM_SEED + i - 1,
            num_iter = 2
        )

        t1 <- proc.time()
        elapsed <- (t1 - t0)["elapsed"]
        times[i] <- elapsed
        n_clusters[i] <- length(unique(partition))
        cat("      Run ", i, "/", n_repeats, ": ",
            sprintf("%.4f", elapsed), "s",
            " (clusters=", n_clusters[i], ")\n", sep = "")
    }

    list(
        times       = times,
        mean_time   = mean(times),
        std_time    = sd(times),
        min_time    = min(times),
        n_clusters  = n_clusters[n_repeats],
        peak_memory_mb = get_memory_mb()
    )
}

# -- Benchmark with igraph::cluster_leiden -----------------------------------
benchmark_igraph_leiden <- function(adj, resolution, objective = "modularity",
                                     n_repeats = N_REPEATS) {
    g <- graph_from_adjacency_matrix(adj, mode = "undirected", weighted = TRUE)

    times <- numeric(n_repeats)
    n_clusters <- integer(n_repeats)
    modularity_vals <- numeric(n_repeats)

    for (i in seq_len(n_repeats)) {
        gc(verbose = FALSE)
        t0 <- proc.time()

        cl <- cluster_leiden(
            g,
            objective_function = objective,
            resolution = resolution,
            weights = E(g)$weight,
            n_iterations = 2
        )

        t1 <- proc.time()
        elapsed <- (t1 - t0)["elapsed"]
        times[i] <- elapsed
        n_clusters[i] <- length(unique(membership(cl)))
        modularity_vals[i] <- modularity(cl)
        cat("      Run ", i, "/", n_repeats, ": ",
            sprintf("%.4f", elapsed), "s",
            " (clusters=", n_clusters[i],
            ", modularity=", sprintf("%.4f", modularity_vals[i]), ")\n", sep = "")
    }

    list(
        times       = times,
        mean_time   = mean(times),
        std_time    = sd(times),
        min_time    = min(times),
        n_clusters  = n_clusters[n_repeats],
        modularity  = mean(modularity_vals),
        peak_memory_mb = get_memory_mb()
    )
}

# -- Main --------------------------------------------------------------------
main <- function() {
    args <- commandArgs(trailingOnly = TRUE)

    if (length(args) > 0) {
        datasets <- args
    } else {
        datasets <- DATASETS
    }

    dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
    all_results <- list()

    for (ds_name in datasets) {
        cat("\n", strrep("=", 60), "\n")
        cat("Dataset:", ds_name, "\n")
        cat(strrep("=", 60), "\n")

        data <- tryCatch(
            load_seurat_from_h5ad(ds_name),
            error = function(e) {
                cat("  [SKIP]", conditionMessage(e), "\n")
                return(NULL)
            }
        )
        if (is.null(data)) next

        adj     <- data$adj
        n_cells <- data$n_cells
        cat("  Benchmarking on", n_cells, "cells...\n")

        for (res in RESOLUTIONS) {
            # --- leidenbase ---
            label <- paste0("seurat_leidenbase_res", res)
            cat("\n  [", label, "] resolution=", res, "\n", sep = "")

            metrics <- tryCatch(
                benchmark_leidenbase(adj, res),
                error = function(e) {
                    cat("    [ERROR]", conditionMessage(e), "\n")
                    return(NULL)
                }
            )

            if (!is.null(metrics)) {
                result <- list(
                    tool            = "seurat",
                    implementation  = "leidenbase",
                    dataset         = ds_name,
                    n_cells         = n_cells,
                    resolution      = res,
                    n_repeats       = N_REPEATS,
                    times           = metrics$times,
                    mean_time       = metrics$mean_time,
                    std_time        = metrics$std_time,
                    min_time        = metrics$min_time,
                    n_clusters      = metrics$n_clusters,
                    modularity      = metrics$modularity %||% NA,
                    peak_memory_mb  = metrics$peak_memory_mb
                )
                all_results[[length(all_results) + 1]] <- result
                save_result(result, paste0(ds_name, "_", label, ".json"))
                cat("    Time:", sprintf("%.3f", metrics$mean_time), "s",
                    "(±", sprintf("%.3f", metrics$std_time), "s)\n")
                cat("    Individual runs:",
                    paste(sprintf("%.4fs", metrics$times), collapse = ", "), "\n")
                cat("    Clusters:", metrics$n_clusters, "\n")
            }

            # --- igraph ---
            label <- paste0("seurat_igraph_res", res)
            cat("\n  [", label, "] resolution=", res, "\n", sep = "")

            metrics <- tryCatch(
                benchmark_igraph_leiden(adj, res),
                error = function(e) {
                    cat("    [ERROR]", conditionMessage(e), "\n")
                    return(NULL)
                }
            )

            if (!is.null(metrics)) {
                result <- list(
                    tool            = "seurat",
                    implementation  = "igraph",
                    dataset         = ds_name,
                    n_cells         = n_cells,
                    resolution      = res,
                    n_repeats       = N_REPEATS,
                    times           = metrics$times,
                    mean_time       = metrics$mean_time,
                    std_time        = metrics$std_time,
                    min_time        = metrics$min_time,
                    n_clusters      = metrics$n_clusters,
                    modularity      = metrics$modularity,
                    peak_memory_mb  = metrics$peak_memory_mb
                )
                all_results[[length(all_results) + 1]] <- result
                save_result(result, paste0(ds_name, "_", label, ".json"))
                cat("    Time:", sprintf("%.3f", metrics$mean_time), "s",
                    "(±", sprintf("%.3f", metrics$std_time), "s)\n")
                cat("    Individual runs:",
                    paste(sprintf("%.4fs", metrics$times), collapse = ", "), "\n")
                cat("    Clusters:", metrics$n_clusters, "\n")
                cat("    Modularity:", sprintf("%.4f", metrics$modularity), "\n")
            }
        }
    }

    # Save all results
    summary_path <- file.path(RESULTS_DIR, "seurat_results.json")
    write(toJSON(all_results, auto_unbox = TRUE, pretty = TRUE), summary_path)
    cat("\nAll Seurat results saved to", summary_path, "\n")
}

main()
