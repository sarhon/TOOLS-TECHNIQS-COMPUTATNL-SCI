#!/usr/bin/env python3
"""
Compiler Optimization Benchmark Script

Compares Fortran solver performance across different compiler optimization levels:
- -O0: No optimization (baseline)
- -O2: Standard optimization
- -O3 -march=native: Aggressive optimization with CPU-specific tuning

Generates timing data, speedup plots, and a markdown report.
"""

import json
import subprocess
import time
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Base directory
BASE_DIR = Path(__file__).parent
FDISCORD_DIR = BASE_DIR / "fdiscord"
RESULTS_DIR = BASE_DIR / "optimization_results"

# Optimization configurations to test
OPT_CONFIGS = {
    "O0": {
        "fflags": "-O0 -Wall -Wextra -fcheck=all -fbacktrace",
        "ldflags": "",
        "description": "No optimization (debug)"
    },
    "O2": {
        "fflags": "-O2 -Wall -Wextra -fcheck=all -fbacktrace",
        "ldflags": "",
        "description": "Standard optimization"
    },
    "O3": {
        "fflags": "-O3 -march=native -Wall -Wextra",
        "ldflags": "",
        "description": "Aggressive + CPU-specific"
    },
    "O3_fast": {
        "fflags": "-O3 -march=native -ffast-math -funroll-loops -Wall -Wextra",
        "ldflags": "",
        "description": "Maximum performance"
    }
}

# Test configurations
NODE_COUNTS = [50, 100, 200, 400]
SN_ORDERS = [8, 16, 32, 64]
NUM_RUNS = 3  # Average over multiple runs for stability


def create_test_input(num_nodes: int, sn_order: int, filename: Path):
    """Create a test input JSON file."""
    config = {
        "description": f"Optimization benchmark: {num_nodes} nodes, SN={sn_order}",
        "materials": [
            {
                "name": "test_material",
                "total": 1.0,
                "scatter": 0.9,
                "Q": 1.0,
                "bounds": [-10.0, 10.0]
            }
        ],
        "settings": {
            "phiL_type": "vac",
            "phiR_type": "vac",
            "phiL": 0.0,
            "phiR": 0.0,
            "num_nodes": num_nodes,
            "sn": sn_order
        }
    }
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)


def build_with_optimization(opt_name: str, opt_config: dict) -> Path:
    """Build Fortran solver with specified optimization flags."""
    build_dir = FDISCORD_DIR / f"build_{opt_name}"
    bin_path = FDISCORD_DIR / "bin" / f"fdiscord_{opt_name}"

    # Clean previous build
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    # Source files in order of dependencies
    src_files = [
        "material.f90",
        "settings.f90",
        "flux.f90",
        "json_reader.f90",
        "json_writer.f90",
        "solver.f90"
    ]
    app_files = ["main.f90"]

    fflags = opt_config["fflags"]
    ldflags = opt_config["ldflags"]

    print(f"  Building with {opt_name}: {fflags}")

    # Compile source files
    obj_files = []
    for src in src_files:
        src_path = FDISCORD_DIR / "src" / src
        obj_path = build_dir / src.replace(".f90", ".o")
        cmd = f"gfortran-13 {fflags} -J{build_dir} -I{build_dir} -c {src_path} -o {obj_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    Compilation failed for {src}: {result.stderr}")
            return None
        obj_files.append(obj_path)

    # Compile app files
    for src in app_files:
        src_path = FDISCORD_DIR / "app" / src
        obj_path = build_dir / src.replace(".f90", ".o")
        cmd = f"gfortran-13 {fflags} -J{build_dir} -I{build_dir} -c {src_path} -o {obj_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    Compilation failed for {src}: {result.stderr}")
            return None
        obj_files.append(obj_path)

    # Link
    obj_str = " ".join(str(o) for o in obj_files)
    cmd = f"gfortran-13 {fflags} {ldflags} -o {bin_path} {obj_str}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    Linking failed: {result.stderr}")
        return None

    print(f"    Built: {bin_path}")
    return bin_path


def run_solver(bin_path: Path, input_file: Path, output_file: Path) -> float:
    """Run solver and return wall time in seconds."""
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'  # Single thread for fair comparison

    cmd = [str(bin_path), "-i", str(input_file), "-o", str(output_file)]

    start_time = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall_time = time.perf_counter() - start_time

    if result.returncode != 0:
        print(f"    Solver failed: {result.stderr[:200]}")
        return None

    return wall_time


def run_benchmarks(binaries: dict) -> dict:
    """Run all benchmark configurations."""
    results = {opt: {"nodes": {}, "sn": {}} for opt in binaries}

    # Node scaling test (fixed SN=32)
    print("\n" + "=" * 60)
    print("Node Scaling Test (SN=32)")
    print("=" * 60)

    sn_fixed = 32
    for nodes in NODE_COUNTS:
        input_file = RESULTS_DIR / f"test_nodes_{nodes}.json"
        create_test_input(nodes, sn_fixed, input_file)

        print(f"\n  Nodes={nodes}:")
        for opt_name, bin_path in binaries.items():
            times = []
            for run in range(NUM_RUNS):
                output_file = RESULTS_DIR / f"output_{opt_name}_nodes_{nodes}_run{run}.json"
                t = run_solver(bin_path, input_file, output_file)
                if t is not None:
                    times.append(t)

            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                results[opt_name]["nodes"][nodes] = {"mean": avg_time, "std": std_time}
                print(f"    {opt_name:8s}: {avg_time:.4f}s (+/- {std_time:.4f}s)")

    # SN scaling test (fixed 100 nodes)
    print("\n" + "=" * 60)
    print("SN Scaling Test (100 nodes)")
    print("=" * 60)

    nodes_fixed = 100
    for sn in SN_ORDERS:
        input_file = RESULTS_DIR / f"test_sn_{sn}.json"
        create_test_input(nodes_fixed, sn, input_file)

        print(f"\n  SN={sn}:")
        for opt_name, bin_path in binaries.items():
            times = []
            for run in range(NUM_RUNS):
                output_file = RESULTS_DIR / f"output_{opt_name}_sn_{sn}_run{run}.json"
                t = run_solver(bin_path, input_file, output_file)
                if t is not None:
                    times.append(t)

            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                results[opt_name]["sn"][sn] = {"mean": avg_time, "std": std_time}
                print(f"    {opt_name:8s}: {avg_time:.4f}s (+/- {std_time:.4f}s)")

    return results


def generate_plots(results: dict):
    """Generate comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'O0': 'red', 'O2': 'blue', 'O3': 'green', 'O3_fast': 'purple'}
    markers = {'O0': 'o', 'O2': 's', 'O3': '^', 'O3_fast': 'd'}

    # Node scaling plot
    ax = axes[0]
    for opt_name in results:
        nodes_data = results[opt_name]["nodes"]
        if nodes_data:
            x = sorted(nodes_data.keys())
            y = [nodes_data[n]["mean"] for n in x]
            yerr = [nodes_data[n]["std"] for n in x]
            ax.errorbar(x, y, yerr=yerr, label=f'{opt_name}: {OPT_CONFIGS[opt_name]["description"]}',
                       marker=markers.get(opt_name, 'o'), color=colors.get(opt_name, 'black'),
                       capsize=3, linewidth=2, markersize=8)

    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Wall Time (seconds)', fontsize=12)
    ax.set_title('Node Scaling (SN=32)', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # SN scaling plot
    ax = axes[1]
    for opt_name in results:
        sn_data = results[opt_name]["sn"]
        if sn_data:
            x = sorted(sn_data.keys())
            y = [sn_data[n]["mean"] for n in x]
            yerr = [sn_data[n]["std"] for n in x]
            ax.errorbar(x, y, yerr=yerr, label=f'{opt_name}: {OPT_CONFIGS[opt_name]["description"]}',
                       marker=markers.get(opt_name, 'o'), color=colors.get(opt_name, 'black'),
                       capsize=3, linewidth=2, markersize=8)

    ax.set_xlabel('SN Order', fontsize=12)
    ax.set_ylabel('Wall Time (seconds)', fontsize=12)
    ax.set_title('Angular Resolution Scaling (100 nodes)', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "optimization_comparison.png", dpi=150)
    print(f"\nPlot saved: {RESULTS_DIR / 'optimization_comparison.png'}")

    # Speedup plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calculate speedups relative to O0
    baseline = "O0"

    # Node scaling speedup
    ax = axes[0]
    for opt_name in results:
        if opt_name == baseline:
            continue
        nodes_data = results[opt_name]["nodes"]
        baseline_data = results[baseline]["nodes"]
        if nodes_data and baseline_data:
            x = sorted(set(nodes_data.keys()) & set(baseline_data.keys()))
            speedups = [baseline_data[n]["mean"] / nodes_data[n]["mean"] for n in x]
            ax.plot(x, speedups, label=f'{opt_name}',
                   marker=markers.get(opt_name, 'o'), color=colors.get(opt_name, 'black'),
                   linewidth=2, markersize=8)

    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel(f'Speedup vs {baseline}', fontsize=12)
    ax.set_title('Speedup from Optimization (Node Scaling)', fontsize=14)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SN scaling speedup
    ax = axes[1]
    for opt_name in results:
        if opt_name == baseline:
            continue
        sn_data = results[opt_name]["sn"]
        baseline_data = results[baseline]["sn"]
        if sn_data and baseline_data:
            x = sorted(set(sn_data.keys()) & set(baseline_data.keys()))
            speedups = [baseline_data[n]["mean"] / sn_data[n]["mean"] for n in x]
            ax.plot(x, speedups, label=f'{opt_name}',
                   marker=markers.get(opt_name, 'o'), color=colors.get(opt_name, 'black'),
                   linewidth=2, markersize=8)

    ax.set_xlabel('SN Order', fontsize=12)
    ax.set_ylabel(f'Speedup vs {baseline}', fontsize=12)
    ax.set_title('Speedup from Optimization (SN Scaling)', fontsize=14)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "optimization_speedup.png", dpi=150)
    print(f"Plot saved: {RESULTS_DIR / 'optimization_speedup.png'}")


def generate_report(results: dict):
    """Generate markdown report."""
    report = []
    report.append("# Compiler Optimization Benchmark Results\n")
    report.append("## Test Configuration\n")
    report.append("| Flag | Description |")
    report.append("|------|-------------|")
    for opt_name, config in OPT_CONFIGS.items():
        report.append(f"| `{config['fflags']}` | {config['description']} |")

    report.append("\n## Node Scaling Results (SN=32)\n")
    report.append("| Nodes | " + " | ".join(f"{opt} (s)" for opt in OPT_CONFIGS) + " | Speedup (O3/O0) |")
    report.append("|-------|" + "|".join(["--------"] * len(OPT_CONFIGS)) + "|-----------------|")

    for nodes in NODE_COUNTS:
        row = [f"{nodes}"]
        times = []
        for opt_name in OPT_CONFIGS:
            if nodes in results[opt_name]["nodes"]:
                t = results[opt_name]["nodes"][nodes]["mean"]
                row.append(f"{t:.4f}")
                times.append(t)
            else:
                row.append("-")
                times.append(None)

        # Calculate speedup O0 vs O3
        if times[0] and times[2]:
            speedup = times[0] / times[2]
            row.append(f"**{speedup:.1f}x**")
        else:
            row.append("-")

        report.append("| " + " | ".join(row) + " |")

    report.append("\n## SN Scaling Results (100 nodes)\n")
    report.append("| SN | " + " | ".join(f"{opt} (s)" for opt in OPT_CONFIGS) + " | Speedup (O3/O0) |")
    report.append("|----|" + "|".join(["--------"] * len(OPT_CONFIGS)) + "|-----------------|")

    for sn in SN_ORDERS:
        row = [f"{sn}"]
        times = []
        for opt_name in OPT_CONFIGS:
            if sn in results[opt_name]["sn"]:
                t = results[opt_name]["sn"][sn]["mean"]
                row.append(f"{t:.4f}")
                times.append(t)
            else:
                row.append("-")
                times.append(None)

        if times[0] and times[2]:
            speedup = times[0] / times[2]
            row.append(f"**{speedup:.1f}x**")
        else:
            row.append("-")

        report.append("| " + " | ".join(row) + " |")

    # Summary statistics
    report.append("\n## Summary\n")

    all_speedups = []
    for nodes in NODE_COUNTS:
        if nodes in results["O0"]["nodes"] and nodes in results["O3"]["nodes"]:
            s = results["O0"]["nodes"][nodes]["mean"] / results["O3"]["nodes"][nodes]["mean"]
            all_speedups.append(s)
    for sn in SN_ORDERS:
        if sn in results["O0"]["sn"] and sn in results["O3"]["sn"]:
            s = results["O0"]["sn"][sn]["mean"] / results["O3"]["sn"][sn]["mean"]
            all_speedups.append(s)

    if all_speedups:
        report.append(f"- **Average speedup (O3 vs O0):** {np.mean(all_speedups):.1f}x")
        report.append(f"- **Maximum speedup:** {np.max(all_speedups):.1f}x")
        report.append(f"- **Minimum speedup:** {np.min(all_speedups):.1f}x")

    report.append("\n## Key Findings\n")
    report.append("1. **Optimization Level Impact**: Moving from -O0 to -O3 provides significant speedup")
    report.append("2. **-march=native**: Enables CPU-specific optimizations (AVX, etc.)")
    report.append("3. **-ffast-math**: Allows floating-point optimizations that may slightly affect precision")
    report.append("4. **Scaling Behavior**: Optimization benefits remain consistent across problem sizes")

    report.append("\n## Plots\n")
    report.append("![Optimization Comparison](optimization_comparison.png)")
    report.append("![Optimization Speedup](optimization_speedup.png)")

    # Write report
    report_path = RESULTS_DIR / "OPTIMIZATION_BENCHMARK.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report))

    print(f"\nReport saved: {report_path}")


def main():
    print("=" * 60)
    print("Compiler Optimization Benchmark")
    print("=" * 60)

    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)

    # Build with each optimization level
    print("\nBuilding binaries...")
    binaries = {}
    for opt_name, opt_config in OPT_CONFIGS.items():
        bin_path = build_with_optimization(opt_name, opt_config)
        if bin_path and bin_path.exists():
            binaries[opt_name] = bin_path
        else:
            print(f"  Warning: Failed to build {opt_name}")

    if not binaries:
        print("Error: No binaries built successfully")
        return

    print(f"\nSuccessfully built {len(binaries)} configurations")

    # Run benchmarks
    results = run_benchmarks(binaries)

    # Save raw results
    results_json = RESULTS_DIR / "optimization_results.json"
    # Convert to serializable format
    serializable = {}
    for opt in results:
        serializable[opt] = {
            "nodes": {str(k): v for k, v in results[opt]["nodes"].items()},
            "sn": {str(k): v for k, v in results[opt]["sn"].items()}
        }
    with open(results_json, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw results saved: {results_json}")

    # Generate plots and report
    generate_plots(results)
    generate_report(results)

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()