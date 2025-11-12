# Fortran Unit Tests

Unit tests for the fdiscord Fortran implementation using pFUnit 4.14.

## Prerequisites

- gfortran-13 (or compatible Fortran compiler)
- pFUnit 4.14 installed

## Setup

Before running the tests, you must set the `PFUNIT_DIR` environment variable to point to your pFUnit installation directory:

```bash
export PFUNIT_DIR=/path/to/pFUnit/install
```

For convenience, add this to your shell configuration file (e.g., `.bashrc` or `.zshrc`).

## Running Tests

Build and run all tests with verbose output:

```bash
make test
```

Or build and run separately:

```bash
make           # Build test executable
./test_runner  # Run tests (quiet mode)
```

### Test Options

The test runner supports several options:

```bash
./test_runner --verbose    # Show detailed test names and progress
./test_runner --filter PATTERN  # Run only tests matching pattern
./test_runner --xml        # Output results in XML format
./test_runner --help       # Show all available options
```

## Build Directory

All compiled artifacts (object files, module files, preprocessed sources) are placed in the `build/` directory to keep the source directory clean. The test executable `test_runner` is created in the main test directory for convenient access.

## Cleaning

Remove all compiled artifacts:

```bash
make clean
```

## Test Files

- `test_gauss_legendre.pf` - Tests for Gauss-Legendre quadrature implementation
- `test_avg_flux.pf` - Tests for average flux calculation with exponential differencing
- `test_flux_solvers.pf` - Tests for tridiagonal solvers used in flux calculations
- `testSuites.inc` - Test suite registration file

## Notes

- `.pf` files are preprocessed to `.F90` files by pFUnit's `funitproc` utility
- Generated files (`.F90`, `.o`, `.mod`) are ignored by git
- Tests link against the source files in `../../fdiscord/src`
