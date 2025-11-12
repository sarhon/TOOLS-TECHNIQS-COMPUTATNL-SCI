# Code Coverage Guide

This document explains how to run tests with code coverage for both Python and Fortran implementations.

## Overview

Code coverage helps identify which parts of your code are being tested and which parts are not. We've set up coverage tools for:
- **Python**: Using `pytest-cov` and `coverage.py`
- **Fortran**: Using `gcov` with gfortran

## Quick Start

### Run All Tests with Coverage

```bash
cd CSE-380/FinalProject/tests
./run_tests_with_coverage.sh
```

This will:
1. Run Python unit tests with coverage
2. Run Python integration tests with coverage
3. Build and run Fortran tests with coverage
4. Generate coverage reports for both languages

## Python Coverage

### Running Python Tests with Coverage

```bash
# From the tests directory
cd CSE-380/FinalProject/tests

# Run specific test file
python -m pytest python/test_pydiscord.py -v --cov=pydiscord --cov-report=term --cov-report=html

# Run all Python tests
python -m pytest python/ -v --cov=pydiscord --cov-report=term --cov-report=html
```

### View Python Coverage Reports

- **Terminal**: Coverage summary is printed automatically
- **HTML**: Open `../coverage_html_report/python/index.html` in a browser

### Python Coverage Configuration

Configuration is in `.coveragerc` at the project root:
- Source directories to measure
- Files to exclude
- HTML report output directory

## Fortran Coverage

### Building with Coverage

```bash
cd CSE-380/FinalProject/fdiscord
make coverage
```

Or for tests:
```bash
cd CSE-380/FinalProject/tests/fortran
make coverage
```

### Running Tests and Generating Coverage

```bash
cd CSE-380/FinalProject/tests/fortran

# Build with coverage
make coverage

# Run tests (generates .gcda files)
./test_fdiscord

# Generate coverage report
make coverage-report
```

### View Fortran Coverage Reports

Coverage reports are in the `coverage/` directory:
- `*.gcov` files contain line-by-line coverage
- `*.gcov.txt` files contain summaries

Example output:
```
File 'solver.f90'
Lines executed:92.50% of 200
```

### Cleaning Coverage Data

```bash
# Python
cd CSE-380/FinalProject
rm -rf .coverage coverage_html_report/

# Fortran
cd CSE-380/FinalProject/fdiscord
make coverage-clean

# Or for tests
cd CSE-380/FinalProject/tests/fortran
make coverage-clean
```

## Understanding Coverage Reports

### Python Coverage Report

```
Name                       Stmts   Miss  Cover
----------------------------------------------
pydiscord/__init__.py          3      0   100%
pydiscord/cli.py              67     67     0%
pydiscord/solver.py          228     29    87%
----------------------------------------------
TOTAL                        298     96    68%
```

- **Stmts**: Total number of statements
- **Miss**: Number of statements not executed
- **Cover**: Percentage of statements executed

### Fortran Coverage Report

```
File 'solver.f90'
Lines executed:87.32% of 213
```

- Shows percentage of lines executed
- `.gcov` files show line-by-line coverage with execution counts

## Makefile Targets

### Main Fortran Makefile

```bash
make coverage         # Build with coverage instrumentation
make coverage-report  # Generate coverage reports
make coverage-clean   # Remove coverage data
```

### Test Fortran Makefile

```bash
make coverage         # Build tests with coverage
make coverage-report  # Generate test coverage reports
make coverage-clean   # Remove coverage data
```

## Coverage Goals

Aim for:
- **>80%** line coverage for core solver code
- **>90%** line coverage for utility functions
- **100%** coverage for critical calculation paths

## Troubleshooting

### Python: Module not found error
```bash
# Install the package in development mode
cd CSE-380/FinalProject
pip install -e .
```

### Python: pytest-cov not installed
```bash
pip install pytest-cov coverage
```

### Fortran: No .gcda files generated
- Make sure you built with `make coverage`
- Run the executable at least once to generate .gcda files

### Fortran: gcov command not found
```bash
# Install gcc/gfortran (gcov is included)
sudo apt-get install gfortran  # On Ubuntu/Debian
```

## Continuous Integration

To integrate coverage into CI/CD:

1. Run tests with coverage in CI pipeline
2. Generate coverage reports
3. Upload to coverage services (Codecov, Coveralls, etc.)
4. Set minimum coverage thresholds

Example GitHub Actions snippet:
```yaml
- name: Run tests with coverage
  run: |
    cd CSE-380/FinalProject/tests
    ./run_tests_with_coverage.sh

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

## Additional Resources

- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [coverage.py documentation](https://coverage.readthedocs.io/)
- [gcov documentation](https://gcc.gnu.org/onlinedocs/gcc/Gcov.html)
- [Fortran code coverage guide](https://gcc.gnu.org/onlinedocs/gcc/Invoking-Gcov.html)