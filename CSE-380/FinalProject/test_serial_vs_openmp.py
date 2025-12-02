#!/usr/bin/env python3
"""
Compare serial (no OpenMP) vs OpenMP with 1 thread to measure OpenMP overhead.
"""

import subprocess
import time
import json
import os
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent
FDISCORD_SERIAL = BASE_DIR / "fdiscord" / "bin" / "fdiscord_serial"
FDISCORD_OPENMP = BASE_DIR / "fdiscord" / "bin" / "fdiscord"

def run_serial(input_file, output_file):
    """Run serial version and return wall time."""
    cmd = [str(FDISCORD_SERIAL), "-i", str(input_file), "-o", str(output_file)]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"Serial failed: {result.stderr}")
        return None
    return elapsed

def run_openmp(input_file, output_file, threads=1):
    """Run OpenMP version and return wall time."""
    cmd = [str(FDISCORD_OPENMP), "-i", str(input_file), "-o", str(output_file)]
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads)
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"OpenMP failed: {result.stderr}")
        return None
    return elapsed

def test_problem(sn_order, num_runs=10):
    """Test a specific problem size."""
    input_file = BASE_DIR / "timing_results_openmp" / f"test_sn_{sn_order}.json"

    if not input_file.exists():
        print(f"Skipping SN={sn_order} (input file not found)")
        return None

    print(f"\nTesting SN={sn_order}, 100 nodes ({num_runs} runs each):")

    # Run serial version
    serial_times = []
    for i in range(num_runs):
        t = run_serial(input_file, f"/tmp/test_serial_{i}.json")
        if t:
            serial_times.append(t)

    # Run OpenMP version with 1 thread
    openmp_times = []
    for i in range(num_runs):
        t = run_openmp(input_file, f"/tmp/test_openmp_{i}.json", threads=1)
        if t:
            openmp_times.append(t)

    if not serial_times or not openmp_times:
        return None

    serial_avg = np.mean(serial_times)
    serial_std = np.std(serial_times)
    openmp_avg = np.mean(openmp_times)
    openmp_std = np.std(openmp_times)

    overhead = ((openmp_avg - serial_avg) / serial_avg) * 100

    print(f"  Serial (no OpenMP):  {serial_avg:.4f}s ± {serial_std:.4f}s")
    print(f"  OpenMP (1 thread):   {openmp_avg:.4f}s ± {openmp_std:.4f}s")
    print(f"  Overhead:            {overhead:+.2f}%")

    return {
        "sn": sn_order,
        "serial_avg": serial_avg,
        "serial_std": serial_std,
        "openmp_avg": openmp_avg,
        "openmp_std": openmp_std,
        "overhead_pct": overhead
    }

def main():
    print("=" * 80)
    print("Serial vs OpenMP (1 thread) Overhead Test")
    print("=" * 80)

    # Test different problem sizes
    test_cases = [32, 64, 96, 128]
    results = []

    for sn in test_cases:
        result = test_problem(sn, num_runs=10)
        if result:
            results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'SN':<6} {'Serial (s)':<12} {'OpenMP (s)':<12} {'Overhead':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['sn']:<6} {r['serial_avg']:<12.4f} {r['openmp_avg']:<12.4f} {r['overhead_pct']:+.2f}%")

    # Overall conclusion
    avg_overhead = np.mean([r['overhead_pct'] for r in results])
    print(f"\nAverage OpenMP overhead: {avg_overhead:+.2f}%")

    if abs(avg_overhead) < 5:
        print("\nConclusion: OpenMP overhead is negligible (< 5%). No benefit to serial build.")
    elif avg_overhead > 5:
        print(f"\nConclusion: OpenMP adds {avg_overhead:.1f}% overhead. Serial build recommended for 1-thread use.")
    else:
        print(f"\nConclusion: Serial is {abs(avg_overhead):.1f}% faster. Serial build recommended for 1-thread use.")

if __name__ == "__main__":
    main()