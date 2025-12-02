# Python Unit Tests

Unit tests for the PyDiscord Python implementation using pytest.

## Prerequisites

- Python 3.7+
- pytest
- numpy
- scipy

Install test dependencies:
```bash
pip install pytest pytest-cov
```

## Running Tests

Run all tests with verbose output:
```bash
pytest tests/python/ -v
```

Run with coverage:
```bash
pytest tests/python/ --cov=pydiscord --cov-report=html
```

## Test Files

### `test_pydiscord.py` - Python Solver Tests (15 tests)

Tests the Python discrete ordinates transport solver:

#### TestGaussLegendre (3 tests)
1. **test_s2_quadrature**: Tests S_2 (N=2) quadrature symmetry and values
2. **test_s4_quadrature**: Tests S_4 quadrature symmetry and weight sum
3. **test_s8_quadrature**: Tests S_8 quadrature symmetry

#### TestMaterial (3 tests)
4. **test_material_creation**: Validates material property initialization
5. **test_material_absorption**: Tests derived absorption cross-section
6. **test_material_bounds**: Tests spatial bounds handling

#### TestSettings (2 tests)
7. **test_settings_vacuum**: Validates vacuum boundary conditions
8. **test_settings_reflective**: Validates reflective boundary conditions

#### TestSingleMaterialVacuum (2 tests)
9. **test_pure_source_no_scatter**: Pure absorption, exponential decay behavior
10. **test_source_with_scatter**: Source with scattering, flux redistribution

#### TestSingleMaterialReflective (1 test)
11. **test_reflective_boundaries**: Validates flux behavior with reflective BCs

#### TestThreeMaterialProblem (1 test)
12. **test_three_material_vacuum**: Multi-region problem with interfaces

#### TestMeshRefinement (1 test)
13. **test_mesh_convergence**: Verifies solution convergence with mesh refinement

#### TestAngularRefinement (1 test)
14. **test_sn_convergence**: Verifies solution convergence with SN order

#### TestConservation (1 test)
15. **test_particle_balance**: Checks particle balance (sources = absorption + leakage)

## Coverage

Generate coverage report:
```bash
pytest tests/python/ --cov=pydiscord --cov-report=html
open htmlcov/index.html
```

## Notes

- Tests use tolerances appropriate for iterative solver results (1e-5 to 1e-6)
- Quadrature tests verify exact values where known analytically
- Conservation tests check particle balance within numerical precision