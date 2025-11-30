#!/bin/bash
# Run all test cases and generate plots

set -e  # Exit on error

CASES=(
    "1_material_ref"
    "2_material_vac"
    "3_material_vac"
    "4_pure_absorption"
    "5_high_scatter"
    "6_reflective_boundaries"
    "7_mixed_boundaries"
    "8_void_penetration"
    "9_dual_source"
    "10_absorption_vs_scatter"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORTRAN_EXE="${SCRIPT_DIR}/fdiscord/bin/fdiscord"

echo "=========================================="
echo "Running all test cases"
echo "=========================================="
echo ""

for case in "${CASES[@]}"; do
    case_dir="${SCRIPT_DIR}/cases/${case}"

    if [ ! -d "$case_dir" ]; then
        echo "⚠️  Skipping $case (directory not found)"
        continue
    fi

    if [ ! -f "$case_dir/input.json" ]; then
        echo "⚠️  Skipping $case (no input.json)"
        continue
    fi

    echo "=========================================="
    echo "Running case: $case"
    echo "=========================================="

    cd "$case_dir"

    # Run Fortran solver
    echo "  → Running Fortran solver..."
    if ! "$FORTRAN_EXE" -i input.json -o foutput.json > /dev/null 2>&1; then
        echo "  ❌ Fortran solver failed!"
        cd "$SCRIPT_DIR"
        continue
    fi
    echo "  ✓ Fortran complete"

    # Run Python solver
    echo "  → Running Python solver..."
    if ! python3 -m pydiscord.cli -i input.json -o pyoutput.json > /dev/null 2>&1; then
        echo "  ❌ Python solver failed!"
        cd "$SCRIPT_DIR"
        continue
    fi
    echo "  ✓ Python complete"

    # Generate plots
    echo "  → Generating plots..."
    if ! python3 ../../postproc.py > /dev/null 2>&1; then
        echo "  ❌ Plotting failed!"
        cd "$SCRIPT_DIR"
        continue
    fi
    echo "  ✓ Plots generated"

    echo "  ✅ Case $case complete!"
    echo ""

    cd "$SCRIPT_DIR"
done

echo "=========================================="
echo "All cases complete!"
echo "=========================================="