#!/usr/bin/env python3
"""
Benchmark GVE-Leiden (OpenMP-based parallel Leiden).

This script:
  1. Clones the GVE-Leiden repo if not present
  2. Compiles with g++ and OpenMP
  3. Converts each dataset's SNN graph (MatrixMarket) to the format GVE-Leiden expects
  4. Runs GVE-Leiden with varying thread counts and parses output
  5. Saves results as JSON

GVE-Leiden repo: https://github.com/puzzlef/leiden-communities-openmp
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
GVE_DIR = os.path.join(PROJECT_DIR, "gve-leiden")

REPO_URL = "https://github.com/puzzlef/leiden-communities-openmp.git"

DATASETS = ["pbmc3k", "pbmc10k", "tabula_sapiens"]
THREAD_COUNTS = [1, 2, 4, 8, 16, 32, 64]
N_REPEATS = 3


def clone_repo():
    """Clone GVE-Leiden if not already present."""
    if os.path.isdir(GVE_DIR) and os.path.exists(os.path.join(GVE_DIR, "main.cxx")):
        print("  GVE-Leiden repo already cloned.")
        return True

    print(f"  Cloning GVE-Leiden from {REPO_URL}...")
    try:
        subprocess.run(
            ["git", "clone", REPO_URL, GVE_DIR],
            check=True, capture_output=True, text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] git clone failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print("  [ERROR] git not found in PATH")
        return False


INSTALL_GUIDE = """
  GVE-Leiden requires OpenMP support for compilation.

  On macOS, install llvm-openmp via conda (recommended):

    conda install -c conda-forge llvm-openmp

  Or via Homebrew:

    brew install libomp

  On Linux, install gcc:

    conda install -c conda-forge gxx
    # or: sudo apt install g++

  After installing, re-run: make benchmark-gveleiden
"""


def _conda_prefix():
    """Return the active conda environment prefix."""
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix and os.path.isdir(prefix):
        return prefix
    for conda_bin in [
        "conda",
        os.path.expanduser("~/miniconda3/bin/conda"),
        os.path.expanduser("~/anaconda3/bin/conda"),
        os.path.expanduser("~/miniforge3/bin/conda"),
        os.path.expanduser("~/mambaforge/bin/conda"),
    ]:
        try:
            r = subprocess.run([conda_bin, "info", "--base"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                base = r.stdout.strip()
                if os.path.isdir(base):
                    return base
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _brew_prefix():
    """Return Homebrew prefix, searching common locations."""
    for candidate in ["/opt/homebrew", "/usr/local", "/home/linuxbrew/.linuxbrew"]:
        if os.path.isfile(os.path.join(candidate, "bin", "brew")):
            return candidate
    try:
        r = subprocess.run(["brew", "--prefix"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _clang_with_omp(omp_prefix):
    """
    Return (compiler, flags) using Apple Clang + an OpenMP library at omp_prefix.
    Adds -Wl,-rpath so dyld can find libomp.dylib at runtime on macOS.
    """
    omp_h   = os.path.join(omp_prefix, "include", "omp.h")
    omp_lib = os.path.join(omp_prefix, "lib")
    libomp  = os.path.join(omp_lib, "libomp.dylib")
    if not os.path.isfile(omp_h) or not os.path.isfile(libomp):
        return None
    flags = [
        "-Xpreprocessor", "-fopenmp",
        f"-I{os.path.join(omp_prefix, 'include')}",
        f"-L{omp_lib}", "-lomp",
        f"-Wl,-rpath,{omp_lib}",   # embed runtime search path in the binary
    ]
    print(f"  Compiler: /usr/bin/c++  (Apple Clang + libomp from {omp_prefix})")
    return "/usr/bin/c++", flags


def _gcc_in_prefix(prefix):
    """Search a prefix/bin for a real g++ (not Apple Clang)."""
    for name in ["g++", "gcc"]:
        cc = os.path.join(prefix, "bin", name)
        try:
            r = subprocess.run([cc, "--version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and "Apple" not in r.stdout:
                print(f"  Compiler: {cc}  ({r.stdout.split(chr(10))[0]})")
                return cc, ["-fopenmp"]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    for ver in range(15, 9, -1):
        cc = os.path.join(prefix, "bin", f"g++-{ver}")
        try:
            r = subprocess.run([cc, "--version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and "Apple" not in r.stdout:
                print(f"  Compiler: {cc}  ({r.stdout.split(chr(10))[0]})")
                return cc, ["-fopenmp"]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return None


def _install_llvm_openmp_conda(conda_prefix):
    """Try to install llvm-openmp into the active conda env."""
    for conda_bin in [
        os.path.join(conda_prefix, "bin", "conda"),
        "conda",
    ]:
        try:
            print(f"  Running: {conda_bin} install -y -c conda-forge llvm-openmp")
            subprocess.run(
                [conda_bin, "install", "-y", "-c", "conda-forge", "llvm-openmp"],
                check=True, timeout=300,
            )
            return True
        except Exception as e:
            print(f"  [WARN] conda install llvm-openmp failed: {e}")
    return False


def detect_compiler():
    """
    Find a C++17 compiler with OpenMP support.

    macOS strategy (Apple Clang + libomp is the most reliable path):
      1. Apple Clang + conda llvm-openmp   (auto-installs if conda active)
      2. Apple Clang + Homebrew libomp
      3. Homebrew / conda gcc (fallback)

    Linux strategy:
      1. conda g++
      2. system g++

    Returns (compiler_path, openmp_flags_list) or (None, None).
    """
    import platform
    is_macos = platform.system() == "Darwin"
    conda    = _conda_prefix()
    brew     = _brew_prefix()

    if is_macos:
        # 1. Apple Clang + conda llvm-openmp  (most reliable on macOS)
        if conda:
            result = _clang_with_omp(conda)
            if result:
                return result
            # Not yet installed — try auto-install
            if _install_llvm_openmp_conda(conda):
                result = _clang_with_omp(conda)
                if result:
                    return result

        # 2. Apple Clang + Homebrew libomp
        if brew:
            libomp = os.path.join(brew, "opt", "libomp")
            result = _clang_with_omp(libomp)
            if result:
                return result

    # Linux (or macOS fallback): real gcc with native -fopenmp
    if conda:
        result = _gcc_in_prefix(conda)
        if result:
            return result
    if brew:
        result = _gcc_in_prefix(brew)
        if result:
            return result
    # Try system PATH
    for ver in range(15, 9, -1):
        try:
            r = subprocess.run([f"g++-{ver}", "--version"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and "Apple" not in r.stdout:
                print(f"  Compiler: g++-{ver}  ({r.stdout.split(chr(10))[0]})")
                return f"g++-{ver}", ["-fopenmp"]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return None, None


def compile_gveleiden(max_threads=64):
    """Compile GVE-Leiden with OpenMP. Returns binary path, or None if unavailable."""
    binary = os.path.join(GVE_DIR, "a.out")
    if os.path.exists(binary):
        print("  GVE-Leiden already compiled.")
        return binary

    compiler, omp_flags = detect_compiler()
    if compiler is None:
        print(INSTALL_GUIDE)
        return None

    defines = [
        "-DTYPE=float",
        f"-DMAX_THREADS={max_threads}",
        "-DREPEAT_METHOD=1",
    ]
    cmd = [
        compiler,
        *defines,
        "-std=c++17", "-O3",
        *omp_flags,
        os.path.join(GVE_DIR, "main.cxx"),
        "-o", binary,
    ]

    print(f"  Compiling: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=GVE_DIR)
        if result.returncode != 0:
            print(f"  [ERROR] Compilation failed:\n{result.stderr}")
            print(INSTALL_GUIDE)
            return None
        print(f"  Compiled successfully: {binary}")
        return binary
    except FileNotFoundError:
        print(f"  [ERROR] Compiler not found: {compiler}")
        print(INSTALL_GUIDE)
        return None


def ensure_symmetric_mtx(mtx_path):
    """
    GVE-Leiden expects a full symmetric MTX file.
    Our export is upper-triangular, so create a full version.
    """
    full_path = mtx_path.replace("_snn.mtx", "_snn_full.mtx")
    if os.path.exists(full_path):
        return full_path

    print(f"  Symmetricizing {mtx_path} -> {full_path}")
    from scipy.io import mmread, mmwrite
    from scipy.sparse import triu

    mat = mmread(mtx_path)
    full = mat + mat.T
    mmwrite(full_path, full)
    return full_path


def parse_gveleiden_output(output):
    """
    Parse GVE-Leiden output line. Expected format:
    {<time>ms, ... <modularity>, <disconnected>} <method>
    """
    results = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue

        m_time = re.search(r"\{(\d+\.?\d*)ms", line)
        m_mod = re.search(r"(\d+\.\d+) modularity", line)
        m_disc = re.search(r"(\d+)/(\d+) disconnected", line)
        m_method = re.search(r"\}\s+(\S+)\s*$", line)

        if m_time:
            entry = {
                "total_time_ms": float(m_time.group(1)),
                "modularity": float(m_mod.group(1)) if m_mod else None,
                "method": m_method.group(1) if m_method else "unknown",
            }
            if m_disc:
                entry["disconnected_communities"] = int(m_disc.group(1))
                entry["total_communities"] = int(m_disc.group(2))
            results.append(entry)
    return results


def run_gveleiden(binary, mtx_path, max_threads=None, symmetric=True):
    """Run GVE-Leiden on a MatrixMarket graph file."""
    env = os.environ.copy()
    if max_threads:
        env["OMP_NUM_THREADS"] = str(max_threads)

    sym_flag = "1" if symmetric else "0"
    weighted_flag = "1"

    cmd = [binary, mtx_path, sym_flag, weighted_flag]
    print(f"    Running: {' '.join(cmd)}")

    try:
        t0 = time.perf_counter()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600, env=env,
            cwd=GVE_DIR,
        )
        t1 = time.perf_counter()

        if result.returncode != 0:
            print(f"    [ERROR] Exit code {result.returncode}")
            if result.stderr:
                print(f"    stderr: {result.stderr[:500]}")
            return None

        wall_time = t1 - t0
        parsed = parse_gveleiden_output(result.stdout)

        leiden_results = [r for r in parsed if "leiden" in r.get("method", "").lower()]
        if leiden_results:
            best = leiden_results[0]
            best["wall_time_s"] = wall_time
            return best

        if parsed:
            parsed[0]["wall_time_s"] = wall_time
            return parsed[0]

        return {"wall_time_s": wall_time, "raw_output": result.stdout[:1000]}

    except subprocess.TimeoutExpired:
        print("    [ERROR] Timed out (>3600s)")
        return None
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark GVE-Leiden")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--threads", nargs="+", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()

    datasets = args.datasets or DATASETS
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("GVE-Leiden Benchmark")
    print("=" * 60)

    if not clone_repo():
        sys.exit(1)

    max_threads = max(args.threads) if args.threads else 64
    binary = compile_gveleiden(max_threads=max_threads)
    if binary is None:
        print("[SKIP] GVE-Leiden compilation not available on this system.")
        print("       Scanpy and Seurat benchmarks are unaffected.")
        sys.exit(0)

    # Auto-detect available cores
    available_cores = os.cpu_count() or 4
    if args.threads:
        thread_counts = args.threads
    else:
        thread_counts = sorted(set(
            t for t in THREAD_COUNTS if t <= available_cores
        ))
        if not thread_counts:
            thread_counts = [1]
    print(f"  Thread counts to test: {thread_counts}")

    all_results = []

    for ds_name in datasets:
        mtx_path = os.path.join(DATA_DIR, f"{ds_name}_snn.mtx")
        if not os.path.exists(mtx_path):
            print(f"\n  [SKIP] {ds_name}: SNN graph not found at {mtx_path}")
            print("  Run benchmark_scanpy.py first to preprocess and export graphs.")
            continue

        full_mtx = ensure_symmetric_mtx(mtx_path)

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        for n_threads in thread_counts:
            print(f"\n  --- {n_threads} thread(s) ---")

            times = []
            for rep in range(args.repeats):
                result = run_gveleiden(binary, full_mtx, max_threads=n_threads)
                if result is None:
                    continue

                t = result.get("total_time_ms", result.get("wall_time_s", 0) * 1000)
                t_sec = t / 1000.0
                times.append(t_sec)
                print(f"      Run {rep+1}/{args.repeats}: {t_sec:.4f}s")

            if not times:
                print("    No successful runs.")
                continue

            import numpy as np
            import psutil
            
            # Estimate peak memory from the process (rough estimate)
            # For GVE-Leiden, we can't directly measure, so use the graph size as proxy
            proc = psutil.Process(os.getpid())
            peak_mem_mb = proc.memory_info().rss / (1024 * 1024)
            
            entry = {
                "tool": "gve-leiden",
                "implementation": f"openmp_{n_threads}t",
                "dataset": ds_name,
                "n_threads": n_threads,
                "n_repeats": len(times),
                "times": times,
                "mean_time": float(np.mean(times)),
                "std_time": float(np.std(times)),
                "min_time": float(np.min(times)),
                "modularity": result.get("modularity"),
                "n_clusters": result.get("total_communities"),
                "peak_memory_mb": peak_mem_mb,
            }
            all_results.append(entry)

            label = f"gveleiden_{n_threads}t"
            result_file = os.path.join(RESULTS_DIR, f"{ds_name}_{label}.json")
            with open(result_file, "w") as f:
                json.dump(entry, f, indent=2)

            print(f"    Time: {np.mean(times):.3f}s (±{np.std(times):.3f}s)")
            print(f"    Individual runs: "
                  + ", ".join(f"{t:.4f}s" for t in times))
            if result.get("modularity"):
                print(f"    Modularity: {result['modularity']:.4f}")
            if result.get("total_communities"):
                print(f"    Communities: {result['total_communities']}")

    summary_path = os.path.join(RESULTS_DIR, "gveleiden_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll GVE-Leiden results saved to {summary_path}")


if __name__ == "__main__":
    main()
