from pydiscord import Material, Settings, solve_flux
import numpy as np
import json
import sys
import os
import argparse


def load_from_json(json_file):
    """Load materials and settings from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Parse materials
    materials = []
    for mat_data in data['materials']:
        materials.append(Material(
            name=mat_data['name'],
            total=mat_data['total'],
            scatter=mat_data['scatter'],
            Q=mat_data['Q'],
            bounds=tuple(mat_data['bounds'])
        ))

    # Parse settings
    settings_data = data['settings']

    # Handle boundary conditions (can be string or float)
    phiL = settings_data.get('phiL_type', 'vac')
    if phiL not in ['vac', 'ref']:
        phiL = settings_data.get('phiL', 0.0)

    phiR = settings_data.get('phiR_type', 'vac')
    if phiR not in ['vac', 'ref']:
        phiR = settings_data.get('phiR', 0.0)

    sim_settings = Settings(
        phiL=phiL,
        phiR=phiR,
        num_nodes=settings_data['num_nodes'],
        sn=settings_data['sn']
    )

    return materials, sim_settings


def write_output_json(output_file, materials, sim_settings, x_edges, x_centers,
                      total_flux, current, angular_flux, mu, w):
    """Write simulation results to JSON file"""
    output_data = {
        "problem_setup": {
            "bounds": [float(x_edges[0]), float(x_edges[-1])],
            "num_nodes": int(sim_settings.num_nodes),
            "sn_order": int(sim_settings.sn),
            "phiL_type": str(sim_settings.phiL),
            "phiR_type": str(sim_settings.phiR)
        },
        "materials": [
            {
                "name": mat.name,
                "total": float(mat.total),
                "scatter": float(mat.scatter),
                "absorption": float(mat.absorption),
                "Q": float(mat.Q),
                "bounds": [float(mat.bounds[0]), float(mat.bounds[1])]
            }
            for mat in materials
        ],
        "quadrature": {
            "mu": mu.tolist(),
            "weights": w.tolist()
        },
        "mesh": {
            "x_edges": x_edges.tolist(),
            "x_centers": x_centers.tolist()
        },
        "solution": {
            "scalar_flux": total_flux.tolist(),
            "current": current.tolist(),
            "angular_flux": angular_flux.tolist()
        },
        "summary": {
            "max_flux": float(np.max(total_flux)),
            "min_flux": float(np.min(total_flux)),
            "flux_at_center": float(total_flux[len(total_flux)//2]),
            "current_at_center": float(current[len(current)//2])
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results written to: {output_file}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='PyDiscord - Python Discrete Ordinates Transport Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-i', '--input',
                       default=os.path.join(os.path.dirname(__file__), 'input.json'),
                       help='Input JSON file (default: ./input.json)')
    parser.add_argument('-o', '--output',
                       default=None,
                       help='Output JSON file (optional)')

    args = parser.parse_args()

    # Load configuration from JSON
    materials, sim_settings = load_from_json(args.input)

    print(f"Loaded configuration from: {args.input}")
    print()

    # Print material information
    print('=' * 41)
    print('Material Configuration:')
    print('=' * 41)
    for i, mat in enumerate(materials, start=1):
        print(f'Material {i}: {mat.name}')
        print(f'  Bounds: [{mat.bounds[0]:8.4f}, {mat.bounds[1]:8.4f}]')
        print(f'  Total XS:      {mat.total:8.4f}')
        print(f'  Scatter XS:    {mat.scatter:8.4f}')
        print(f'  Absorption XS: {mat.absorption:8.4f}')
        print(f'  Source Q:      {mat.Q:8.4f}')
        print()

    # Print settings information
    print('=' * 41)
    print('Solver Settings:')
    print('=' * 41)
    print(f'  Left BC:    {sim_settings.phiL}')
    print(f'  Right BC:   {sim_settings.phiR}')
    print(f'  Num Nodes:  {sim_settings.num_nodes:5d}')
    print(f'  SN Order:   {sim_settings.sn:5d}')
    print()

    # Solve the flux
    x_edges, total_flux, current, angular_flux, mu, w, x_centers = solve_flux(materials, sim_settings)

    # Print some results
    print('=' * 41)
    print('Solution Summary:')
    print('=' * 41)
    mid_idx = sim_settings.num_nodes // 2
    print(f'  Total flux at x=0:   {total_flux[mid_idx]:12.4E}')
    print(f'  Current at x=0:      {current[mid_idx]:12.4E}')
    print(f'  Max flux:            {np.max(total_flux):12.4E}')
    print(f'  Min flux:            {np.min(total_flux):12.4E}')
    print()

    # Write output JSON if requested
    if args.output:
        write_output_json(args.output, materials, sim_settings, x_edges, x_centers,
                         total_flux, current, angular_flux, mu, w)


if __name__ == '__main__':
    main()