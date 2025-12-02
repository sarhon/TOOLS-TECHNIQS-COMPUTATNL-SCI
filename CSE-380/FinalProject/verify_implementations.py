#!/usr/bin/env python3
"""
Implementation Verification Script

Compares Python and Fortran solver outputs across all test cases.
Reports maximum differences and validates numerical agreement.

Generates:
- Console summary of verification results
- verification_results/VERIFICATION_REPORT.md
- Per-case difference statistics
"""

import json
import subprocess
import os
from pathlib import Path, PosixPath
import numpy as np

# Directories
BASE_DIR = Path(__file__).parent
CASES_DIR = BASE_DIR / "cases"
FDISCORD_BIN = BASE_DIR / "fdiscord" / "bin" / "fdiscord"
RESULTS_DIR = BASE_DIR / "verification_results"

# Tolerance for "identical" results
RTOL = 1e-10  # Relative tolerance
ATOL = 1e-14  # Absolute tolerance


def find_test_cases():
    """Find all test case directories with input.json files."""
    cases = []
    for case_dir in sorted(CASES_DIR.iterdir()):
        if case_dir.is_dir():
            input_file = case_dir / "input.json"
            if input_file.exists():
                cases.append(case_dir)
    return cases


def run_python_solver(input_file: Path, output_file: Path) -> bool:
    """Run Python solver and return success status."""
    cmd = ["python3", "-m", "pydiscord.cli", "-i", str(input_file), "-o", str(output_file)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    return result.returncode == 0


def run_fortran_solver(input_file: Path, output_file: Path) -> bool:
    """Run Fortran solver and return success status."""
    if not FDISCORD_BIN.exists():
        print(f"Error: Fortran binary not found at {FDISCORD_BIN}")
        print("Run 'cd fdiscord && make' to build it")
        return False

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'

    cmd = [str(FDISCORD_BIN), "-i", str(input_file), "-o", str(output_file)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.returncode == 0


def load_output(output_file: Path) -> dict:
    """Load solver output JSON file."""
    with open(output_file, 'r') as f:
        return json.load(f)


def compare_arrays(py_arr, f_arr, name: str) -> dict:
    """Compare two arrays and return statistics."""
    py_arr = np.array(py_arr)
    f_arr = np.array(f_arr)

    if py_arr.shape != f_arr.shape:
        return {
            "name": name,
            "status": "SHAPE_MISMATCH",
            "py_shape": py_arr.shape,
            "f_shape": f_arr.shape
        }

    abs_diff = np.abs(py_arr - f_arr)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    # Relative difference (avoiding division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.abs((py_arr - f_arr) / np.where(np.abs(py_arr) > ATOL, py_arr, 1.0))
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0.0)

    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)

    # Check if arrays are "identical" within tolerance
    identical = np.allclose(py_arr, f_arr, rtol=RTOL, atol=ATOL)

    return {
        "name": name,
        "status": "PASS" if identical else "DIFFER",
        "max_abs_diff": float(max_abs_diff),
        "mean_abs_diff": float(mean_abs_diff),
        "max_rel_diff": float(max_rel_diff),
        "mean_rel_diff": float(mean_rel_diff),
        "identical": identical
    }


def compare_outputs(py_output: dict, f_output: dict) -> dict:
    """Compare Python and Fortran outputs comprehensively."""
    results = {
        "arrays": [],
        "scalars": {},
        "overall_pass": True
    }

    # Compare scalar flux
    if "scalar_flux" in py_output and "scalar_flux" in f_output:
        comp = compare_arrays(py_output["scalar_flux"], f_output["scalar_flux"], "scalar_flux")
        results["arrays"].append(comp)
        if not comp.get("identical", False):
            results["overall_pass"] = False

    # Compare current
    if "current" in py_output and "current" in f_output:
        comp = compare_arrays(py_output["current"], f_output["current"], "current")
        results["arrays"].append(comp)
        if not comp.get("identical", False):
            results["overall_pass"] = False

    # Compare angular flux (if present)
    if "angular_flux" in py_output and "angular_flux" in f_output:
        comp = compare_arrays(py_output["angular_flux"], f_output["angular_flux"], "angular_flux")
        results["arrays"].append(comp)
        if not comp.get("identical", False):
            results["overall_pass"] = False

    # Compare summary scalars
    if "summary" in py_output and "summary" in f_output:
        py_sum = py_output["summary"]
        f_sum = f_output["summary"]

        for key in ["flux_at_zero", "current_at_zero", "max_flux", "min_flux"]:
            if key in py_sum and key in f_sum:
                py_val = py_sum[key]
                f_val = f_sum[key]
                diff = abs(py_val - f_val)
                rel_diff = diff / abs(py_val) if abs(py_val) > ATOL else diff
                results["scalars"][key] = {
                    "python": py_val,
                    "fortran": f_val,
                    "abs_diff": diff,
                    "rel_diff": rel_diff,
                    "pass": rel_diff < RTOL
                }

    # Compare convergence
    if "convergence" in py_output and "convergence" in f_output:
        py_conv = py_output["convergence"]
        f_conv = f_output["convergence"]

        if "iterations" in py_conv and "iterations" in f_conv:
            results["scalars"]["iterations"] = {
                "python": py_conv["iterations"],
                "fortran": f_conv["iterations"],
                "match": py_conv["iterations"] == f_conv["iterations"]
            }

    return results


def run_verification():
    """Run verification on all test cases."""
    cases = find_test_cases()
    print(f"Found {len(cases)} test cases\n")

    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = {}
    summary_lines = []

    print("=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)

    for case_dir in cases:
        case_name = case_dir.name
        input_file = case_dir / "input.json"
        py_output = RESULTS_DIR / f"{case_name}_py.json"
        f_output = RESULTS_DIR / f"{case_name}_f.json"

        print(f"\n{case_name}:")
        print("-" * 40)

        # Run both solvers
        py_success = run_python_solver(input_file, py_output)
        f_success = run_fortran_solver(input_file, f_output)

        if not py_success:
            print("  Python solver FAILED")
            all_results[case_name] = {"status": "PYTHON_FAILED"}
            summary_lines.append(f"| {case_name} | FAIL | Python solver failed |")
            continue

        if not f_success:
            print("  Fortran solver FAILED")
            all_results[case_name] = {"status": "FORTRAN_FAILED"}
            summary_lines.append(f"| {case_name} | FAIL | Fortran solver failed |")
            continue

        # Load and compare outputs
        py_data = load_output(py_output)
        f_data = load_output(f_output)

        comparison = compare_outputs(py_data, f_data)
        all_results[case_name] = comparison

        # Print results
        status = "PASS" if comparison["overall_pass"] else "DIFFER"
        print(f"  Status: {status}")

        max_diff = 0.0
        for arr_comp in comparison["arrays"]:
            name = arr_comp["name"]
            if "max_rel_diff" in arr_comp:
                diff = arr_comp["max_rel_diff"]
                max_diff = max(max_diff, diff)
                status_str = "✓" if arr_comp.get("identical", False) else "✗"
                print(f"  {name:15s}: max_rel_diff = {diff:.2e} {status_str}")

        if comparison["scalars"].get("iterations"):
            iter_info = comparison["scalars"]["iterations"]
            match_str = "✓" if iter_info.get("match", False) else "✗"
            print(f"  Iterations: Py={iter_info['python']}, F={iter_info['fortran']} {match_str}")

        summary_lines.append(f"| {case_name} | {status} | max_rel_diff = {max_diff:.2e} |")

    # Generate report
    generate_report(all_results, summary_lines, cases)

    return all_results


def generate_report(results: dict, summary_lines: list, cases: list):
    """Generate markdown verification report."""
    report = []
    report.append("# Implementation Verification Report\n")
    report.append("Comparison of Python and Fortran solver outputs.\n")

    report.append("## Summary\n")
    report.append("| Case | Status | Max Relative Difference |")
    report.append("|------|--------|------------------------|")
    report.extend(summary_lines)

    # Statistics
    pass_count = sum(1 for r in results.values()
                     if isinstance(r, dict) and r.get("overall_pass", False))
    total_count = len(cases)

    report.append(f"\n**Passed: {pass_count}/{total_count}**\n")

    # Detailed results
    report.append("## Detailed Results\n")

    for case_name, result in results.items():
        report.append(f"### {case_name}\n")

        if not isinstance(result, dict):
            report.append(f"Error: {result}\n")
            continue

        if result.get("status") in ["PYTHON_FAILED", "FORTRAN_FAILED"]:
            report.append(f"**{result['status']}**\n")
            continue

        # Array comparisons
        if "arrays" in result:
            report.append("**Array Comparisons:**\n")
            report.append("| Array | Max Abs Diff | Max Rel Diff | Status |")
            report.append("|-------|--------------|--------------|--------|")

            for arr in result["arrays"]:
                if "max_abs_diff" in arr:
                    status = "✓ PASS" if arr.get("identical", False) else "✗ DIFFER"
                    report.append(
                        f"| {arr['name']} | {arr['max_abs_diff']:.2e} | {arr['max_rel_diff']:.2e} | {status} |"
                    )

        # Scalar comparisons
        if "scalars" in result and result["scalars"]:
            report.append("\n**Scalar Comparisons:**\n")
            report.append("| Value | Python | Fortran | Rel Diff |")
            report.append("|-------|--------|---------|----------|")

            for key, val in result["scalars"].items():
                if "python" in val and "fortran" in val:
                    report.append(
                        f"| {key} | {val['python']:.6e} | {val['fortran']:.6e} | {val.get('rel_diff', 0):.2e} |"
                    )

        report.append("")

    # Conclusions
    report.append("## Conclusions\n")

    if pass_count == total_count:
        report.append("All test cases passed verification. Both implementations produce")
        report.append("numerically identical results within floating-point precision.\n")
        report.append(f"- Tolerance: rtol={RTOL}, atol={ATOL}")
        report.append("- Maximum observed relative difference: < 1e-10")
    else:
        report.append(f"{total_count - pass_count} test case(s) showed differences.")
        report.append("Review the detailed results above.\n")

    report.append("\n## Verification Method\n")
    report.append("1. Run both Python and Fortran solvers on identical input")
    report.append("2. Compare output arrays element-by-element")
    report.append("3. Calculate absolute and relative differences")
    report.append(f"4. Pass if: `numpy.allclose(py, f, rtol={RTOL}, atol={ATOL})`")

    # Write report
    report_path = RESULTS_DIR / "VERIFICATION_REPORT.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report))

    print(f"\nReport saved: {report_path}")

    # Save raw results
    results_json = RESULTS_DIR / "verification_results.json"
    # Convert numpy types to Python types for JSON serialization
    serializable = {}
    for case, res in results.items():
        if isinstance(res, dict):
            serializable[case] = {
                k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                for k, v in res.items()
            }
        else:
            serializable[case] = str(res)

    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Raw results: {results_json}")


def main():
    print("=" * 70)
    print("Python vs Fortran Implementation Verification")
    print("=" * 70)

    # Check prerequisites
    if not FDISCORD_BIN.exists():
        print(f"\nError: Fortran binary not found at {FDISCORD_BIN}")
        print("Building Fortran solver...")
        result = subprocess.run(["make"], cwd=BASE_DIR / "fdiscord", capture_output=True)
        if result.returncode != 0:
            print("Build failed. Please run 'cd fdiscord && make' manually.")
            return

    results = run_verification()

    # Final summary
    print("\n" + "=" * 70)
    pass_count = sum(1 for r in results.values()
                     if isinstance(r, dict) and r.get("overall_pass", False))
    total = len(results)

    if pass_count == total:
        print(f"VERIFICATION COMPLETE: All {total} cases PASSED")
    else:
        print(f"VERIFICATION COMPLETE: {pass_count}/{total} cases passed")
    print("=" * 70)


if __name__ == "__main__":
    main()