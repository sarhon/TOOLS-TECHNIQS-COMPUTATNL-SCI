#!/usr/bin/env python3
"""
Timing comparison script for Fortran vs Python discrete ordinates solver.

Tests three configurations:
1. Fortran (no MPI)
2. Fortran with MPI (1, 2, 4 cores)
3. Python (no MPI)

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
FDISCORD_MPI_BIN = BASE_DIR / "fdiscord" / "bin" / "fdiscord_mpi"
TIMING_DIR = BASE_DIR / "timing_results"

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
                "scatter": 0.9,  # High scattering for realistic iteration count
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


def run_fortran(input_file, output_file, omp_threads=None):
    """Run Fortran solver and return wall time in seconds."""
    cmd = [str(FDISCORD_BIN), "-i", str(input_file), "-o", str(output_file)]

    env = os.environ.copy()
    if omp_threads:
        env['OMP_NUM_THREADS'] = str(omp_threads)
    else:
        env['OMP_NUM_THREADS'] = '1'  # Default to serial

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
        "fortran_serial": [],
        "fortran_omp2": [],
        "fortran_omp4": [],
        "fortran_omp8": [],
        "python": []
    }

    for num_nodes in node_counts:
        print(f"\nTesting with {num_nodes} nodes, SN={sn_order}...")

        # Create test input
        input_file = TIMING_DIR / f"test_nodes_{num_nodes}.json"
        create_test_input(num_nodes, sn_order, input_file)

        # Fortran serial (1 thread)
        print("  Running Fortran serial (1 thread)...", end=" ", flush=True)
        output_file = TIMING_DIR / f"output_f_nodes_{num_nodes}.json"
        t = run_fortran(input_file, output_file, omp_threads=1)
        results["fortran_serial"].append(t)
        print(f"{t:.3f}s" if t else "FAILED")

        # Fortran OpenMP (2 threads)
        print("  Running Fortran OpenMP (2 threads)...", end=" ", flush=True)
        output_file = TIMING_DIR / f"output_fomp2_nodes_{num_nodes}.json"
        t = run_fortran(input_file, output_file, omp_threads=2)
        results["fortran_omp2"].append(t)
        print(f"{t:.3f}s" if t else "FAILED")

        # Fortran OpenMP (4 threads)
        print("  Running Fortran OpenMP (4 threads)...", end=" ", flush=True)
        output_file = TIMING_DIR / f"output_fomp4_nodes_{num_nodes}.json"
        t = run_fortran(input_file, output_file, omp_threads=4)
        results["fortran_omp4"].append(t)
        print(f"{t:.3f}s" if t else "FAILED")

        # Fortran OpenMP (8 threads)
        print("  Running Fortran OpenMP (8 threads)...", end=" ", flush=True)
        output_file = TIMING_DIR / f"output_fomp8_nodes_{num_nodes}.json"
        t = run_fortran(input_file, output_file, omp_threads=8)
        results["fortran_omp8"].append(t)
        print(f"{t:.3f}s" if t else "FAILED")

        # Python
        print("  Running Python...", end=" ", flush=True)
        output_file = TIMING_DIR / f"output_py_nodes_{num_nodes}.json"
        t = run_python(input_file, output_file)
        results["python"].append(t)
        print(f"{t:.3f}s" if t else "FAILED")

    # Save results
    with open(TIMING_DIR / "node_scaling_results.json", 'w') as f:
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
        "fortran": [],
        "fortran_mpi_1": [],
        "fortran_mpi_2": [],
        "fortran_mpi_4": [],
        "python": []
    }

    for sn_order in sn_orders:
        print(f"\nTesting with nodes={num_nodes}, SN={sn_order}...")

        # Create test input
        input_file = TIMING_DIR / f"test_sn_{sn_order}.json"
        create_test_input(num_nodes, sn_order, input_file)

        # Fortran (no MPI)
        print("  Running Fortran (no MPI)...", end=" ", flush=True)
        output_file = TIMING_DIR / f"output_f_sn_{sn_order}.json"
        t = run_fortran(input_file, output_file)
        results["fortran"].append(t)
        print(f"{t:.3f}s" if t else "FAILED")

        # Fortran MPI (1 proc)
        print("  Running Fortran MPI (1 proc)...", end=" ", flush=True)
        output_file = TIMING_DIR / f"output_fmpi1_sn_{sn_order}.json"
        t = run_fortran(input_file, output_file, mpi_procs=1)
        results["fortran_mpi_1"].append(t)
        print(f"{t:.3f}s" if t else "FAILED")

        # Fortran MPI (2 procs)
        print("  Running Fortran MPI (2 procs)...", end=" ", flush=True)
        output_file = TIMING_DIR / f"output_fmpi2_sn_{sn_order}.json"
        t = run_fortran(input_file, output_file, mpi_procs=2)
        results["fortran_mpi_2"].append(t)
        print(f"{t:.3f}s" if t else "FAILED")

        # Fortran MPI (4 procs)
        print("  Running Fortran MPI (4 procs)...", end=" ", flush=True)
        output_file = TIMING_DIR / f"output_fmpi4_sn_{sn_order}.json"
        t = run_fortran(input_file, output_file, mpi_procs=4)
        results["fortran_mpi_4"].append(t)
        print(f"{t:.3f}s" if t else "FAILED")

        # Python
        print("  Running Python...", end=" ", flush=True)
        output_file = TIMING_DIR / f"output_py_sn_{sn_order}.json"
        t = run_python(input_file, output_file)
        results["python"].append(t)
        print(f"{t:.3f}s" if t else "FAILED")

    # Save results
    with open(TIMING_DIR / "sn_scaling_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def plot_results(node_results, sn_results):
    """Generate comparison plots."""
    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Node scaling
    ax1.plot(node_results["nodes"], node_results["fortran"],
             'o-', label='Fortran (no MPI)', linewidth=2, markersize=8)
    ax1.plot(node_results["nodes"], node_results["fortran_mpi_1"],
             's-', label='Fortran MPI (1 proc)', linewidth=2, markersize=8)
    ax1.plot(node_results["nodes"], node_results["fortran_mpi_2"],
             'd-', label='Fortran MPI (2 procs)', linewidth=2, markersize=8)
    ax1.plot(node_results["nodes"], node_results["fortran_mpi_4"],
             '^-', label='Fortran MPI (4 procs)', linewidth=2, markersize=8)
    ax1.plot(node_results["nodes"], node_results["python"],
             'x-', label='Python', linewidth=2, markersize=10)

    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('Wall Time (seconds)', fontsize=12)
    ax1.set_title('Scaling with Number of Nodes (SN=32)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Plot 2: SN scaling
    ax2.plot(sn_results["sn"], sn_results["fortran"],
             'o-', label='Fortran (no MPI)', linewidth=2, markersize=8)
    ax2.plot(sn_results["sn"], sn_results["fortran_mpi_1"],
             's-', label='Fortran MPI (1 proc)', linewidth=2, markersize=8)
    ax2.plot(sn_results["sn"], sn_results["fortran_mpi_2"],
             'd-', label='Fortran MPI (2 procs)', linewidth=2, markersize=8)
    ax2.plot(sn_results["sn"], sn_results["fortran_mpi_4"],
             '^-', label='Fortran MPI (4 procs)', linewidth=2, markersize=8)
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

    # Create speedup plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Speedup for node scaling (relative to Python)
    python_times = np.array(node_results["python"])
    ax1.plot(node_results["nodes"],
             python_times / np.array(node_results["fortran"]),
             'o-', label='Fortran (no MPI)', linewidth=2, markersize=8)
    ax1.plot(node_results["nodes"],
             python_times / np.array(node_results["fortran_mpi_1"]),
             's-', label='Fortran MPI (1 proc)', linewidth=2, markersize=8)
    ax1.plot(node_results["nodes"],
             python_times / np.array(node_results["fortran_mpi_2"]),
             'd-', label='Fortran MPI (2 procs)', linewidth=2, markersize=8)
    ax1.plot(node_results["nodes"],
             python_times / np.array(node_results["fortran_mpi_4"]),
             '^-', label='Fortran MPI (4 procs)', linewidth=2, markersize=8)
    ax1.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='Python baseline')

    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('Speedup vs Python', fontsize=12)
    ax1.set_title('Speedup Comparison (SN=32)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Speedup for SN scaling (relative to Python)
    python_times = np.array(sn_results["python"])
    ax2.plot(sn_results["sn"],
             python_times / np.array(sn_results["fortran"]),
             'o-', label='Fortran (no MPI)', linewidth=2, markersize=8)
    ax2.plot(sn_results["sn"],
             python_times / np.array(sn_results["fortran_mpi_1"]),
             's-', label='Fortran MPI (1 proc)', linewidth=2, markersize=8)
    ax2.plot(sn_results["sn"],
             python_times / np.array(sn_results["fortran_mpi_2"]),
             'd-', label='Fortran MPI (2 procs)', linewidth=2, markersize=8)
    ax2.plot(sn_results["sn"],
             python_times / np.array(sn_results["fortran_mpi_4"]),
             '^-', label='Fortran MPI (4 procs)', linewidth=2, markersize=8)
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
    print(f"{'Nodes':<10} {'Fortran':<12} {'F-MPI(1)':<12} {'F-MPI(2)':<12} {'F-MPI(4)':<12} {'Python':<12}")
    print("-" * 80)
    for i, n in enumerate(node_results["nodes"]):
        print(f"{n:<10} {node_results['fortran'][i]:<12.3f} "
              f"{node_results['fortran_mpi_1'][i]:<12.3f} "
              f"{node_results['fortran_mpi_2'][i]:<12.3f} "
              f"{node_results['fortran_mpi_4'][i]:<12.3f} "
              f"{node_results['python'][i]:<12.3f}")

    print("\nSN Scaling (100 nodes):")
    print(f"{'SN':<10} {'Fortran':<12} {'F-MPI(1)':<12} {'F-MPI(2)':<12} {'F-MPI(4)':<12} {'Python':<12}")
    print("-" * 80)
    for i, sn in enumerate(sn_results["sn"]):
        print(f"{sn:<10} {sn_results['fortran'][i]:<12.3f} "
              f"{sn_results['fortran_mpi_1'][i]:<12.3f} "
              f"{sn_results['fortran_mpi_2'][i]:<12.3f} "
              f"{sn_results['fortran_mpi_4'][i]:<12.3f} "
              f"{sn_results['python'][i]:<12.3f}")

    # Average speedups
    avg_speedup_f = np.mean(np.array(node_results["python"]) / np.array(node_results["fortran"]))
    avg_speedup_fmpi4 = np.mean(np.array(node_results["python"]) / np.array(node_results["fortran_mpi_4"]))

    print(f"\nAverage Speedup (Fortran vs Python): {avg_speedup_f:.2f}x")
    print(f"Average Speedup (Fortran MPI 4-proc vs Python): {avg_speedup_fmpi4:.2f}x")


def main():
    """Main execution."""
    print("Discrete Ordinates Solver - Timing Comparison")
    print("=" * 80)
    print(f"Results will be saved to: {TIMING_DIR}")
    print()

    # Run tests
    node_results = test_node_scaling()
    sn_results = test_sn_scaling()

    # Generate plots
    plot_results(node_results, sn_results)

    # Print summary
    print_summary(node_results, sn_results)

    print("\n" + "=" * 80)
    print("Timing comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()