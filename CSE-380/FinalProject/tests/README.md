# FinalProject Test Suite

This directory contains comprehensive unit and integration tests for both Python and Fortran implementations of the discrete ordinates transport solver.

## Test Structure

```
tests/
├── python/                      # Python unit tests
│   └── test_pydiscord.py       # PyDiscord unit tests
├── fortran/                     # Fortran unit tests
│   ├── test_fdiscord.f90       # FDiscord unit tests
│   └── Makefile                # Build system for Fortran tests
├── test_cases/                  # Test input files
│   ├── single_material_vac.json
│   ├── single_material_ref.json
│   └── three_material_vac.json
├── test_integration.py          # Integration tests comparing Python vs Fortran
├── run_all_tests.sh            # Master test runner script
└── README.md                    # This file
```

## Running Tests

### Run All Tests

```bash
cd tests
./run_all_tests.sh
```

This will run:
1. Python unit tests
2. Fortran unit tests
3. Integration tests comparing Python and Fortran outputs

### Run Individual Test Suites

**Python Unit Tests:**
```bash
cd tests
python -m pytest python/test_pydiscord.py -v
```

**Fortran Unit Tests:**
```bash
cd tests/fortran
make clean
make
./test_fdiscord
```

**Integration Tests:**
```bash
cd tests
python -m pytest test_integration.py -v
```

## Test Coverage

### Python Unit Tests (`python/test_pydiscord.py`)

- **TestGaussLegendre**: Gauss-Legendre quadrature generation
  - S2, S4, S8 quadrature accuracy
  - Symmetry properties
  - Weight normalization

- **TestMaterial**: Material class functionality
  - Material creation
  - Absorption calculation
  - Boundary validation

- **TestSettings**: Settings class functionality
  - Vacuum boundary conditions
  - Reflective boundary conditions

- **TestSingleMaterialVacuum**: Single material with vacuum BC
  - Pure source without scattering
  - Source with scattering
  - Array shape validation

- **TestSingleMaterialReflective**: Single material with reflective BC
  - Reflective boundary validation
  - Current at boundaries

- **TestThreeMaterialProblem**: Multi-material problems
  - Three material scatter-source-scatter configuration
  - Flux symmetry
  - Known solution validation

- **TestMeshRefinement**: Mesh convergence studies
  - Convergence with mesh refinement

- **TestAngularRefinement**: SN order studies
  - Convergence with angular refinement

- **TestConservation**: Conservation properties
  - Particle balance validation

### Fortran Unit Tests (`fortran/test_fdiscord.f90`)

- **test_gauss_legendre**: Gauss-Legendre quadrature
  - S2, S4, S8 accuracy and symmetry
  - Weight normalization

- **test_avg_flux_function**: Average flux calculation
  - Small tau Taylor series
  - Flux decay without source

- **test_single_material_vacuum**: Single material vacuum BC
  - Flux positivity
  - Symmetry validation
  - Peak location

- **test_single_material_reflective**: Single material reflective BC
  - Current validation at boundaries

- **test_three_material_vacuum**: Three material problem
  - Known solution validation
  - Max/min flux values

- **test_symmetry**: Flux symmetry
  - Symmetric problems produce symmetric solutions

### Integration Tests (`test_integration.py`)

- **TestPythonFortranComparison**: Direct comparison of solvers
  - Single material vacuum BC
  - Single material reflective BC
  - Three material problem
  - Quadrature agreement

- **TestParameterSpace**: Parameter space exploration
  - Different mesh sizes (5, 10, 20 nodes)
  - Different SN orders (S2, S4, S8)
  - Different scattering ratios (0.0, 0.5, 0.9, 0.99)

## Test Cases

### `single_material_vac.json`
- Single source material
- Vacuum boundaries
- 10 nodes, S4

### `single_material_ref.json`
- Single source material with high scattering
- Reflective boundaries
- 10 nodes, S4

### `three_material_vac.json`
- Three materials: scatter-source-scatter
- Vacuum boundaries
- 27 nodes, S8
- Matches the main input.json problem

## Requirements

### Python
- Python 3.6+
- pytest
- numpy

Install with:
```bash
pip install pytest numpy
```

### Fortran
- gfortran (or compatible Fortran compiler)
- make

## Expected Results

All tests should pass with the following tolerances:
- Quadrature values: 1e-10 relative tolerance
- Flux comparisons: 1e-5 relative tolerance
- Conservation: 5% absolute tolerance

## Adding New Tests

### Python Tests

Add new test classes or methods to `python/test_pydiscord.py`:

```python
class TestNewFeature:
    """Test description"""

    def test_specific_case(self):
        """Test specific case"""
        # Setup
        materials = [...]
        settings = Settings(...)

        # Solve
        x_edges, total_flux, ... = solve_flux(materials, settings)

        # Assert
        assert condition
```

### Fortran Tests

Add new test subroutines to `fortran/test_fdiscord.f90`:

```fortran
subroutine test_new_feature(passed, failed, total)
    integer, intent(inout) :: passed, failed, total

    ! Setup

    ! Test
    call assert_true(condition, 'test name', passed, failed, total)

    ! Cleanup
end subroutine test_new_feature
```

Then call it from the main program.

### Integration Tests

Add new test methods to `test_integration.py`:

```python
def test_new_comparison(self):
    """Test description"""
    input_file = 'test_cases/new_case.json'

    # Run both
    py_results = self.run_python(input_file)
    fortran_output = self.run_fortran(input_file, output_file)

    # Compare
    self.compare_solutions(py_results, fortran_output)
```

## Continuous Integration

These tests are designed to be compatible with CI/CD pipelines. The `run_all_tests.sh` script returns:
- Exit code 0: All tests passed
- Exit code 1: One or more tests failed

Example GitHub Actions integration (future work):
```yaml
- name: Run tests
  run: |
    cd tests
    ./run_all_tests.sh
```