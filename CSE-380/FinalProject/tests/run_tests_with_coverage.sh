#!/bin/bash

# Run all tests with code coverage for FinalProject

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "========================================="
echo "Running All Tests with Coverage"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test suite
run_test() {
    local test_name=$1
    local test_command=$2

    echo -e "${YELLOW}Running: ${test_name}${NC}"
    echo "----------------------------------------"

    if eval "$test_command"; then
        echo -e "${GREEN}✓ PASSED: ${test_name}${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAILED: ${test_name}${NC}"
        ((TESTS_FAILED++))
    fi
    echo ""
}

# Clean previous coverage data
echo -e "${BLUE}Cleaning previous coverage data...${NC}"
cd "$PROJECT_DIR"
rm -rf .coverage coverage_html_report htmlcov
cd "$SCRIPT_DIR/fortran"
make coverage-clean > /dev/null 2>&1 || true
echo ""

# ============================================
# PYTHON TESTS WITH COVERAGE
# ============================================
echo -e "${BLUE}=== Python Tests with Coverage ===${NC}"
echo ""

# Run Python unit tests with coverage
run_test "Python Unit Tests (with coverage)" \
    "cd $SCRIPT_DIR && python -m pytest python/test_pydiscord.py -v --cov=pydiscord --cov-report=term --cov-report=html:../coverage_html_report/python"

# Run integration tests with coverage
run_test "Integration Tests (with coverage)" \
    "cd $SCRIPT_DIR && python -m pytest test_integration.py -v --cov=pydiscord --cov-append --cov-report=term --cov-report=html:../coverage_html_report/python"

# Generate Python coverage summary
echo -e "${BLUE}=== Python Coverage Summary ===${NC}"
cd "$PROJECT_DIR"
python -m coverage report --rcfile=.coveragerc 2>/dev/null || echo "Coverage report generation failed"
echo ""
echo -e "${GREEN}Python HTML coverage report: ${PROJECT_DIR}/coverage_html_report/python/index.html${NC}"
echo ""

# ============================================
# FORTRAN TESTS WITH COVERAGE
# ============================================
echo -e "${BLUE}=== Fortran Tests with Coverage ===${NC}"
echo ""

# Build Fortran tests with coverage
echo -e "${YELLOW}Building Fortran Tests with coverage...${NC}"
cd "$SCRIPT_DIR/fortran"
if make coverage > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Fortran tests built with coverage${NC}"
    echo ""

    # Run Fortran tests
    run_test "Fortran Unit Tests (with coverage)" \
        "cd $SCRIPT_DIR/fortran && ./test_fdiscord"

    # Generate coverage report
    echo -e "${BLUE}=== Fortran Coverage Summary ===${NC}"
    cd "$SCRIPT_DIR/fortran"
    make coverage-report
    echo ""
    echo -e "${GREEN}Fortran coverage reports: ${SCRIPT_DIR}/fortran/coverage/${NC}"
    echo ""
else
    echo -e "${RED}✗ FAILED: Fortran tests failed to build${NC}"
    ((TESTS_FAILED++))
fi

# ============================================
# SUMMARY
# ============================================
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"
echo ""

echo "========================================="
echo "Coverage Reports"
echo "========================================="
echo -e "Python:  ${PROJECT_DIR}/coverage_html_report/python/index.html"
echo -e "Fortran: ${SCRIPT_DIR}/fortran/coverage/"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests PASSED!${NC}"
    exit 0
else
    echo -e "${RED}Some tests FAILED!${NC}"
    exit 1
fi
