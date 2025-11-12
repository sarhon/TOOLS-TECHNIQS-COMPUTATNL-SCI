# Fortran Unit Tests

[![Fortran Tests](https://github.com/sarhon/TOOLS-TECHNIQS-COMPUTATNL-SCI/actions/workflows/fortran-tests.yml/badge.svg)](https://github.com/sarhon/TOOLS-TECHNIQS-COMPUTATNL-SCI/actions/workflows/fortran-tests.yml)
[![codecov](https://codecov.io/gh/sarhon/TOOLS-TECHNIQS-COMPUTATNL-SCI/branch/main/graph/badge.svg)](https://codecov.io/gh/sarhon/TOOLS-TECHNIQS-COMPUTATNL-SCI)

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

## Code Coverage

### Text Coverage Reports

Generate text-based code coverage reports for the tested source files:

```bash
make coverage
```

This will:
1. Rebuild the tests with coverage instrumentation
2. Run all tests
3. Generate coverage reports in the `coverage/` directory
4. Display a summary of line coverage for each source file

Coverage reports are in `.gcov` format and show which lines were executed during testing.

### HTML Coverage Reports

Generate browsable HTML coverage reports:

```bash
make html-coverage
```

This will:
1. Rebuild the tests with coverage instrumentation
2. Run all tests
3. Generate interactive HTML coverage reports in `coverage/html/`
4. Provide overall and per-file coverage statistics with line-by-line highlighting

Open `coverage/html/index.html` in a web browser to view the detailed coverage report with color-coded source files showing which lines were executed.

## Cleaning

Remove all compiled artifacts:

```bash
make clean
```

This removes build files, test executables, test results, and coverage reports.

## Continuous Integration

This test suite runs automatically on GitHub Actions for every push and pull request to the `main` and `develop` branches. The CI pipeline:

1. **Builds pFUnit from source** and caches it for faster subsequent runs
2. **Compiles the tests** with gfortran-13
3. **Runs all tests** with verbose output
4. **Generates coverage reports** (both text and HTML)
5. **Uploads coverage** to Codecov (requires `CODECOV_TOKEN` secret)
6. **Publishes HTML coverage** as a downloadable artifact

### Setting up CI

To enable CI on your fork:

1. **Update badge URLs** in this README by replacing `YOUR_USERNAME` with your GitHub username
2. **Optional: Enable Codecov**
   - Sign up at [codecov.io](https://codecov.io) and link your repository
   - Add `CODECOV_TOKEN` to your repository secrets (Settings → Secrets → Actions)
   - Without this token, coverage upload will be skipped but won't fail the CI

### Viewing CI Results

- **Test status**: Click the badge at the top of this README
- **Coverage reports**: Download from the "Artifacts" section of any workflow run
- **Test output**: View in the "Run tests" step of the workflow logs

## Test Files

The test suite rigorously validates the core functionality of the neutron transport solver. Each test file focuses on a specific module or component:

### `test_material.pf` - Material Module Tests (5 tests)

Tests the Material type and its initialization, focusing on physical properties and calculations:

1. **test_material_initialization**: Verifies that all material properties (name, total cross-section, scatter cross-section, source term, spatial bounds) are correctly initialized
2. **test_material_absorption_calculation**: Validates the derived absorption cross-section (σ_a = σ_t - σ_s)
3. **test_material_pure_absorber**: Tests edge case of zero scattering (pure absorber material)
4. **test_material_pure_scatterer**: Tests edge case of zero absorption (σ_s = σ_t, pure scatterer)
5. **test_material_multiple_instances**: Ensures multiple material instances maintain independent properties

### `test_settings.pf` - Settings Module Tests (6 tests)

Tests the Settings/Configuration type for simulation parameters and boundary conditions:

1. **test_settings_initialization**: Validates correct initialization of all configuration fields
2. **test_settings_reflective_boundaries**: Tests reflective boundary condition setup (zero net current)
3. **test_settings_fixed_boundaries**: Tests fixed/Dirichlet boundary conditions with specified flux values
4. **test_settings_mixed_boundaries**: Tests heterogeneous boundaries (different types on left/right)
5. **test_settings_node_count**: Validates spatial discretization parameter handling
6. **test_settings_quadrature_orders**: Tests different Sn quadrature orders (S2, S4, S8, etc.)

### `test_flux_copy.pf` - Flux Type Copy Semantics Tests (6 tests)

Tests copy constructor and assignment operators for RightFlux and LeftFlux types, ensuring proper memory management:

1. **test_right_flux_copy_constructor**: Validates deep copy of all RightFlux fields (scalars and arrays)
2. **test_left_flux_copy_constructor**: Validates deep copy of all LeftFlux fields
3. **test_right_flux_independent_after_copy**: Ensures copied RightFlux objects are independent (no shallow copy issues)
4. **test_left_flux_independent_after_copy**: Ensures copied LeftFlux objects are independent
5. **test_right_flux_assignment_chain**: Tests multiple consecutive assignments (rf3 = rf2 = rf1)
6. **test_left_flux_assignment_chain**: Tests chained assignments for LeftFlux

These tests prevent memory corruption bugs and ensure proper Fortran derived type semantics.

### `test_solve_flux.pf` - Solver Integration Tests (7 tests)

Integration tests for the complete neutron transport solver. These test the full solve_flux routine under various physical conditions:

1. **test_solve_flux_single_material_vacuum**: Basic solver validation with homogeneous material, vacuum boundaries, and S4 quadrature. Verifies domain setup, positive flux, and symmetry for a simple problem.
2. **test_solve_flux_fixed_boundaries**: Tests non-zero Dirichlet boundary conditions (incoming flux specified). Ensures boundary values influence the solution appropriately.
3. **test_solve_flux_reflective_boundaries**: Validates reflective BCs by checking near-zero net current at boundaries, important for symmetric problems.
4. **test_solve_flux_multiple_materials**: Tests heterogeneous problems with material interfaces, verifying flux continuity across discontinuous cross-sections.
5. **test_solve_flux_pure_absorber**: Tests convergence with no scattering (σ_s = 0), a challenging case with no neutron feedback between directions.
6. **test_solve_flux_optional_outputs**: Validates that optional output arrays (angular flux, quadrature points μ, weights w, cell centers) are correctly allocated, sized, and populated with valid data.
7. **test_solve_flux_high_scattering**: Tests convergence with high scattering ratio (c = σ_s/σ_t = 0.99), a numerically challenging regime requiring many source iterations.

These integration tests ensure the solver produces physically meaningful results (positive flux, energy conservation, boundary condition satisfaction) and handles edge cases robustly.

### `testSuites.inc` - Test Suite Registration

pFUnit configuration file that registers all test modules for execution. Automatically updated when new test modules are added.

## Notes

- `.pf` files are preprocessed to `.F90` files by pFUnit's `funitproc` utility
- Generated files (`.F90`, `.o`, `.mod`) are placed in the `build/` directory
- Tests link against the source files in `../../fdiscord/src`
- All tests use appropriate numerical tolerances (typically 1.0e-10 for exact comparisons, 1.0e-5 to 1.0e-6 for iterative solver results)

test change