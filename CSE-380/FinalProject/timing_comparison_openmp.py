#!/usr/bin/env python3
"""
OpenMP Timing comparison for Fortran discrete ordinates solver.

Tests:
1. Fortran Serial (1 thread)
2. Fortran OpenMP (2, 4, 8 threads)
3. Python

Measures scaling with:
- Increasing number of nodes
- Increasing SN order
"""

import json
import subprocess
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Base directory
BASE_DIR = Path(__file__).parent
FDISCORD_BIN = BASE_DIR / "fdiscord" / "bin" / "fdiscord"
TIMING_DIR = BASE_DIR / "timing_results_openmp"

# Ensure timing directory exists
TIMING_DIR.mkdir(exist_ok=True)


def create_test_input(num_nodes, sn_order, filename):
    """Create a test input JSON file."""
    config = {
        "description": "Timing test case",
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


def run_fortran(input_file, output_file, omp_threads=1):
    """Run Fortran solver and return wall time in seconds."""
    cmd = [str(FDISCORD_BIN), "-i", str(input_file), "-o", str(output_file)]

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(omp_threads)

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall_time = time.time() - start_time

    if result.returncode != 0:
        print(f"Fortran solver failed: {result.stderr}")
        return None

    return wall_time


def run_python(input_file, output_file):
    """Run Python solver and return wall time in seconds."""
    cmd = ["python3", "-m", "pydiscord.cli", "-i", str(input_file), "-o", str(output_file)]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    wall_time = time.time() - start_time

    if result.returncode != 0:
        print(f"Python solver failed: {result.stderr}")
        return None

    return wall_time


def test_node_scaling():
    """Test scaling with increasing number of nodes."""
    print("=" * 80)
    print("Testing Node Scaling (fixed SN=32)")
    print("=" * 80)

    sn_order = 32
    node_counts = [50, 100, 200, 400, 800]

    results = {
        "nodes": node_counts,
        "fortran_1t": [],
        "fortran_2t": [],
        "fortran_4t": [],
        "fortran_8t": [],
        "python": []
    }

    for num_nodes in node_counts:
        print(f"\nTesting with {num_nodes} nodes, SN={sn_order}...")

        input_file = TIMING_DIR / f"test_nodes_{num_nodes}.json"
        create_test_input(num_nodes, sn_order, input_file)

        for threads, key in [(1, "fortran_1t"), (2, "fortran_2t"), (4, "fortran_4t"), (8, "fortran_8t")]:
            print(f"  Fortran ({threads} thread{'s' if threads > 1 else ''})...", end=" ", flush=True)
            output_file = TIMING_DIR / f"out_f{threads}t_n{num_nodes}.json"
            t = run_fortran(input_file, output_file, omp_threads=threads)
            results[key].append(t)
            print(f"{t:.3f}s" if t else "FAILED")

        print("  Python...", end=" ", flush=True)
        output_file = TIMING_DIR / f"out_py_n{num_nodes}.json"
        t = run_python(input_file, output_file)
        results["python"].append(t)
        print(f"{t:.3f}s" if t else "FAILED")

    with open(TIMING_DIR / "node_scaling.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def test_sn_scaling():
    """Test scaling with increasing SN order."""
    print("\n" + "=" * 80)
    print("Testing SN Scaling (fixed nodes=100)")
    print("=" * 80)

    num_nodes = 100
    sn_orders = [8, 16, 32, 64, 96, 128]

    results = {
        "sn": sn_orders,
        "fortran_1t": [],
        "fortran_2t": [],
        "fortran_4t": [],
        "fortran_8t": [],
        "python": []
    }

    for sn_order in sn_orders:
        print(f"\nTesting with nodes={num_nodes}, SN={sn_order}...")

        input_file = TIMING_DIR / f"test_sn_{sn_order}.json"
        create_test_input(num_nodes, sn_order, input_file)

        for threads, key in [(1, "fortran_1t"), (2, "fortran_2t"), (4, "fortran_4t"), (8, "fortran_8t")]:
            print(f"  Fortran ({threads} thread{'s' if threads > 1 else ''})...", end=" ", flush=True)
            output_file = TIMING_DIR / f"out_f{threads}t_sn{sn_order}.json"
            t = run_fortran(input_file, output_file, omp_threads=threads)
            results[key].append(t)
            print(f"{t:.3f}s" if t else "FAILED")

        print("  Python...", end=" ", flush=True)
        output_file = TIMING_DIR / f"out_py_sn{sn_order}.json"
        t = run_python(input_file, output_file)
        results["python"].append(t)
        print(f"{t:.3f}s" if t else "FAILED")

    with open(TIMING_DIR / "sn_scaling.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def plot_results(node_results, sn_results):
    """Generate comparison plots."""
    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Node scaling
    ax1.plot(node_results["nodes"], node_results["fortran_1t"],
             'o-', label='Fortran (1 thread)', linewidth=2, markersize=8)
    ax1.plot(node_results["nodes"], node_results["fortran_2t"],
             's-', label='Fortran (2 threads)', linewidth=2, markersize=8)
    ax1.plot(node_results["nodes"], node_results["fortran_4t"],
             'd-', label='Fortran (4 threads)', linewidth=2, markersize=8)
    ax1.plot(node_results["nodes"], node_results["fortran_8t"],
             '^-', label='Fortran (8 threads)', linewidth=2, markersize=8)
    ax1.plot(node_results["nodes"], node_results["python"],
             'x-', label='Python', linewidth=2, markersize=10)

    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('Wall Time (seconds)', fontsize=12)
    ax1.set_title('Scaling with Number of Nodes (SN=32)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # SN scaling
    ax2.plot(sn_results["sn"], sn_results["fortran_1t"],
             'o-', label='Fortran (1 thread)', linewidth=2, markersize=8)
    ax2.plot(sn_results["sn"], sn_results["fortran_2t"],
             's-', label='Fortran (2 threads)', linewidth=2, markersize=8)
    ax2.plot(sn_results["sn"], sn_results["fortran_4t"],
             'd-', label='Fortran (4 threads)', linewidth=2, markersize=8)
    ax2.plot(sn_results["sn"], sn_results["fortran_8t"],
             '^-', label='Fortran (8 threads)', linewidth=2, markersize=8)
    ax2.plot(sn_results["sn"], sn_results["python"],
             'x-', label='Python', linewidth=2, markersize=10)

    ax2.set_xlabel('SN Order', fontsize=12)
    ax2.set_ylabel('Wall Time (seconds)', fontsize=12)
    ax2.set_title('Scaling with SN Order (100 nodes)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(TIMING_DIR / 'timing_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {TIMING_DIR / 'timing_comparison.png'}")

    # Speedup plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Node scaling speedup
    python_times = np.array(node_results["python"])
    ax1.plot(node_results["nodes"],
             python_times / np.array(node_results["fortran_1t"]),
             'o-', label='Fortran (1 thread)', linewidth=2, markersize=8)
    ax1.plot(node_results["nodes"],
             python_times / np.array(node_results["fortran_2t"]),
             's-', label='Fortran (2 threads)', linewidth=2, markersize=8)
    ax1.plot(node_results["nodes"],
             python_times / np.array(node_results["fortran_4t"]),
             'd-', label='Fortran (4 threads)', linewidth=2, markersize=8)
    ax1.plot(node_results["nodes"],
             python_times / np.array(node_results["fortran_8t"]),
             '^-', label='Fortran (8 threads)', linewidth=2, markersize=8)
    ax1.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='Python baseline')

    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('Speedup vs Python', fontsize=12)
    ax1.set_title('Speedup Comparison (SN=32)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # SN scaling speedup
    python_times = np.array(sn_results["python"])
    ax2.plot(sn_results["sn"],
             python_times / np.array(sn_results["fortran_1t"]),
             'o-', label='Fortran (1 thread)', linewidth=2, markersize=8)
    ax2.plot(sn_results["sn"],
             python_times / np.array(sn_results["fortran_2t"]),
             's-', label='Fortran (2 threads)', linewidth=2, markersize=8)
    ax2.plot(sn_results["sn"],
             python_times / np.array(sn_results["fortran_4t"]),
             'd-', label='Fortran (4 threads)', linewidth=2, markersize=8)
    ax2.plot(sn_results["sn"],
             python_times / np.array(sn_results["fortran_8t"]),
             '^-', label='Fortran (8 threads)', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='Python baseline')

    ax2.set_xlabel('SN Order', fontsize=12)
    ax2.set_ylabel('Speedup vs Python', fontsize=12)
    ax2.set_title('Speedup Comparison (100 nodes)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig(TIMING_DIR / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {TIMING_DIR / 'speedup_comparison.png'}")


def print_summary(node_results, sn_results):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nNode Scaling (SN=32):")
    print(f"{'Nodes':<10} {'F-1t':<10} {'F-2t':<10} {'F-4t':<10} {'F-8t':<10} {'Python':<10}")
    print("-" * 70)
    for i, n in enumerate(node_results["nodes"]):
        print(f"{n:<10} {node_results['fortran_1t'][i]:<10.3f} "
              f"{node_results['fortran_2t'][i]:<10.3f} "
              f"{node_results['fortran_4t'][i]:<10.3f} "
              f"{node_results['fortran_8t'][i]:<10.3f} "
              f"{node_results['python'][i]:<10.3f}")

    print("\nSN Scaling (100 nodes):")
    print(f"{'SN':<10} {'F-1t':<10} {'F-2t':<10} {'F-4t':<10} {'F-8t':<10} {'Python':<10}")
    print("-" * 70)
    for i, sn in enumerate(sn_results["sn"]):
        print(f"{sn:<10} {sn_results['fortran_1t'][i]:<10.3f} "
              f"{sn_results['fortran_2t'][i]:<10.3f} "
              f"{sn_results['fortran_4t'][i]:<10.3f} "
              f"{sn_results['fortran_8t'][i]:<10.3f} "
              f"{sn_results['python'][i]:<10.3f}")

    # OpenMP speedups
    avg_speedup_1t = np.mean(np.array(node_results["fortran_1t"]) / np.array(node_results["fortran_1t"]))
    avg_speedup_2t = np.mean(np.array(node_results["fortran_1t"]) / np.array(node_results["fortran_2t"]))
    avg_speedup_4t = np.mean(np.array(node_results["fortran_1t"]) / np.array(node_results["fortran_4t"]))
    avg_speedup_8t = np.mean(np.array(node_results["fortran_1t"]) / np.array(node_results["fortran_8t"]))

    print("\nOpenMP Speedup (relative to 1 thread):")
    print(f"  2 threads: {avg_speedup_2t:.2f}x")
    print(f"  4 threads: {avg_speedup_4t:.2f}x")
    print(f"  8 threads: {avg_speedup_8t:.2f}x")

    # Python comparison
    avg_vs_python = np.mean(np.array(node_results["python"]) / np.array(node_results["fortran_4t"]))
    print(f"\nFortran (4 threads) vs Python: {avg_vs_python:.2f}x faster")


def main():
    """Main execution."""
    print("Discrete Ordinates Solver - OpenMP Timing Comparison")
    print("=" * 80)
    print(f"Results will be saved to: {TIMING_DIR}")
    print()

    node_results = test_node_scaling()
    sn_results = test_sn_scaling()

    plot_results(node_results, sn_results)
    print_summary(node_results, sn_results)

    print("\n" + "=" * 80)
    print("Timing comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()