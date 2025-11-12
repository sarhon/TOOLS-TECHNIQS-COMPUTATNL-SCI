#!/bin/bash

# Run all tests for FinalProject

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "========================================="
echo "Running All FinalProject Tests"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# 1. Run Python unit tests
run_test "Python Unit Tests" "cd $SCRIPT_DIR && python -m pytest python/test_pydiscord.py -v"

# 2. Build and run Fortran unit tests
echo -e "${YELLOW}Building Fortran Tests...${NC}"
cd "$SCRIPT_DIR/fortran"
make clean > /dev/null 2>&1
if make > /dev/null 2>&1; then
    run_test "Fortran Unit Tests" "cd $SCRIPT_DIR/fortran && ./test_fdiscord"
else
    echo -e "${RED}✗ FAILED: Fortran tests failed to build${NC}"
    ((TESTS_FAILED++))
fi
echo ""

# 3. Run integration tests (Python vs Fortran)
run_test "Integration Tests (Python vs Fortran)" "cd $SCRIPT_DIR && python -m pytest test_integration.py -v"

# Print summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests PASSED!${NC}"
    exit 0
else
    echo -e "${RED}Some tests FAILED!${NC}"
    exit 1
fi