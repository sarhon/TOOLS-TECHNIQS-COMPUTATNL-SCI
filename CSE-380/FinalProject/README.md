# Discrete Ordinates Transport Solver

A 1D discrete ordinates (SN) transport solver implemented in both Python and Fortran, with unified JSON input for consistency.

## Project Structure

```
FinalProject/
├── input.json              # Unified input configuration
├── Python/
│   ├── PyDiscord.py        # Python solver library
│   └── pdiscord.py         # Python executable
└── Fortran/
    ├── src/                # Library modules
    │   ├── material.f90    # Material data structures
    │   ├── settings.f90    # Solver settings
    │   ├── flux.f90        # Flux solvers (left/right going)
    │   ├── json_reader.f90 # JSON input parser
    │   └── solver.f90      # Main solver implementation
    ├── app/                # Application main programs
    │   └── main.f90        # Main program entry point
    ├── bin/                # Final executables (created by build)
    │   └── fdiscord        # Fortran executable
    ├── build/              # Build artifacts (*.o, *.mod)
    └── Makefile            # Build system
```

## Building

### Python
No build required. Dependencies:
- numpy
- scipy
- matplotlib

### Fortran
```bash
cd Fortran
make           # Build the project
make clean     # Remove build artifacts
make rebuild   # Clean and rebuild
make run       # Build and run
make help      # Show available targets
```

## Running

### Python
```bash
cd Python
python pdiscord.py [path/to/input.json]
```

### Fortran
```bash
cd Fortran
./bin/fdiscord [path/to/input.json]
```

Both executables will use `../input.json` by default if no path is provided.

## Input Format

The `input.json` file specifies materials and solver settings:

```json
{
  "description": "Problem description",
  "materials": [
    {
      "name": "material1",
      "total": 2.0,
      "scatter": 1.99,
      "Q": 0.0,
      "bounds": [-15.0, -5.0]
    }
  ],
  "settings": {
    "phiL_type": "vac",
    "phiR_type": "vac",
    "phiL": 0.0,
    "phiR": 0.0,
    "num_nodes": 27,
    "sn": 8
  }
}
```

### Material Properties
- `name`: Material identifier
- `total`: Total cross section (σt)
- `scatter`: Scattering cross section (σs)
- `Q`: Isotropic source strength
- `bounds`: Spatial extent [left, right]

### Solver Settings
- `phiL_type`, `phiR_type`: Boundary condition type ("vac", "ref", or "fixed")
- `phiL`, `phiR`: Boundary flux values (used if type is "fixed")
- `num_nodes`: Number of spatial mesh cells
- `sn`: Quadrature order (number of discrete directions)

## Output

Both solvers output:
- Material configuration
- Solver settings
- Iteration convergence history
- Final solution summary:
  - Total flux at x=0
  - Current at x=0
  - Maximum and minimum flux values

## Verification

Both implementations produce identical results when run with the same input configuration, ensuring correctness across languages.

## Example Results

```
Solution Summary:
  Total flux at x=0:     9.9974E-01
  Current at x=0:       -1.2514E-04
  Max flux:              9.9974E-01
  Min flux:              2.9369E-02
```