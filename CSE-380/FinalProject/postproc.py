import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def load_output_json(filepath):
    """Load output JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_scalar_flux(data_dict, output_dir=None):
    """
    Plot scalar flux vs position for multiple datasets.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with keys as labels and values as data dictionaries
    output_dir : Path or str, optional
        Directory to save plots
    """
    plt.figure(figsize=(10, 6))

    for label, data in data_dict.items():
        x_edges = np.array(data['mesh']['x_edges'])
        scalar_flux = np.array(data['solution']['scalar_flux'])
        plt.plot(x_edges, scalar_flux, '-o', label=label, markersize=4)

    plt.xlabel('Position (cm)', fontsize=12)
    plt.ylabel('Scalar Flux', fontsize=12)
    plt.title('Scalar Flux Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'scalar_flux.png', dpi=600)
        print(f"Saved scalar flux plot to {output_dir / 'scalar_flux.png'}")

    plt.close()


def plot_current(data_dict, output_dir=None):
    """
    Plot current vs position for multiple datasets.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with keys as labels and values as data dictionaries
    output_dir : Path or str, optional
        Directory to save plots
    """
    plt.figure(figsize=(10, 6))

    for label, data in data_dict.items():
        x_edges = np.array(data['mesh']['x_edges'])
        current = np.array(data['solution']['current'])
        plt.plot(x_edges, current, '-o', label=label, markersize=4)

    plt.xlabel('Position (cm)', fontsize=12)
    plt.ylabel('Current', fontsize=12)
    plt.title('Current Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'current.png', dpi=600)
        print(f"Saved current plot to {output_dir / 'current.png'}")

    plt.close()


def plot_angular_flux_selected_angles(data_dict, angle_indices=None, output_dir=None):
    """
    Plot angular flux for selected angle indices.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with keys as labels and values as data dictionaries
    angle_indices : list of int, optional
        List of angle indices to plot. If None, plots a subset of angles
    output_dir : Path or str, optional
        Directory to save plots
    """
    # Get first dataset to determine default angles
    first_data = list(data_dict.values())[0]
    mu_values = np.array(first_data['quadrature']['mu'])
    n_angles = len(mu_values)

    if angle_indices is None:
        # Select subset of angles: some negative, near-zero, and positive
        angle_indices = [0, n_angles//8, n_angles//4, n_angles//2-1,
                        n_angles//2, n_angles//2+n_angles//8,
                        3*n_angles//4, n_angles-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Plot 4 selected angles
    for i, angle_idx in enumerate(angle_indices[:4]):
        ax = axes[i]
        mu = mu_values[angle_idx]

        for label, data in data_dict.items():
            x_edges = np.array(data['mesh']['x_edges'])
            angular_flux = np.array(data['solution']['angular_flux'])
            flux_at_angle = angular_flux[angle_idx, :]
            ax.plot(x_edges, flux_at_angle, '-o', label=label, markersize=3)

        ax.set_xlabel('Position (cm)', fontsize=10)
        ax.set_ylabel('Angular Flux', fontsize=10)
        ax.set_title(f'Angular Flux at μ = {mu:.4f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'angular_flux_selected.png', dpi=600)
        print(f"Saved selected angular flux plot to {output_dir / 'angular_flux_selected.png'}")

    plt.close()


def plot_angular_flux_heatmap(data, label="", output_dir=None):
    """
    Plot angular flux as a heatmap (angle vs position).

    Parameters:
    -----------
    data : dict
        Data dictionary with solution and mesh information
    label : str
        Label for the dataset
    output_dir : Path or str, optional
        Directory to save plots
    """
    x_edges = np.array(data['mesh']['x_edges'])
    mu_values = np.array(data['quadrature']['mu'])
    angular_flux = np.array(data['solution']['angular_flux'])

    plt.figure(figsize=(12, 8))
    im = plt.imshow(angular_flux, aspect='auto', origin='lower',
                    extent=[x_edges[0], x_edges[-1], mu_values[0], mu_values[-1]],
                    cmap='viridis')

    plt.colorbar(im, label='Angular Flux')
    plt.xlabel('Position (cm)', fontsize=12)
    plt.ylabel('Direction Cosine (μ)', fontsize=12)
    plt.title(f'Angular Flux Heatmap - {label}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        safe_label = label.replace(' ', '_').lower()
        plt.savefig(output_dir / f'angular_flux_heatmap_{safe_label}.png', dpi=600)
        print(f"Saved angular flux heatmap to {output_dir / f'angular_flux_heatmap_{safe_label}.png'}")

    plt.close()


def plot_all_angular_fluxes(data, label="", output_dir=None):
    """
    Plot all angular fluxes with scalar flux envelope and material regions.

    Parameters:
    -----------
    data : dict
        Data dictionary with solution and mesh information
    label : str
        Label for the dataset
    output_dir : Path or str, optional
        Directory to save plots
    """
    x_edges = np.array(data['mesh']['x_edges'])
    mu_values = np.array(data['quadrature']['mu'])
    angular_flux = np.array(data['solution']['angular_flux'])
    scalar_flux = np.array(data['solution']['scalar_flux'])

    # Get material information
    materials = data.get('materials', [])

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create colormap for angular fluxes
    n_angles = len(mu_values)
    colors_angular = plt.cm.rainbow(np.linspace(0, 1, n_angles))

    # Plot all angular fluxes
    for i in range(n_angles):
        ax.plot(x_edges, angular_flux[i, :], '-',
                color=colors_angular[i], linewidth=1.5, alpha=0.7)

    # Plot scalar flux as thick black line on top
    ax.plot(x_edges, scalar_flux, 'ko-', linewidth=2, markersize=4,
            label='Scalar Flux', zorder=100)

    # Plot material regions as background shading (after plotting to get proper ylim)
    colors = ['lightblue', 'peachpuff', 'lightgreen', 'lavender', 'lightyellow']
    ymax = ax.get_ylim()[1]
    for i, mat in enumerate(materials):
        bounds = mat['bounds']
        color = colors[i % len(colors)]
        ax.axvspan(bounds[0], bounds[1], alpha=0.3, color=color, zorder=0)
        # Add material name label
        center = (bounds[0] + bounds[1]) / 2
        ax.text(center, ymax * 0.95, mat['name'],
                ha='center', va='top', fontsize=10, style='italic', zorder=101)

    ax.set_xlabel('Position [cm]', fontsize=14)
    ax.set_ylabel('Fluence [cm$^{-2}$]', fontsize=14)
    ax.set_title(f'Angular Flux Distribution - {label}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend(fontsize=12)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        safe_label = label.replace(' ', '_').lower()
        plt.savefig(output_dir / f'all_angular_flux_{safe_label}.png', dpi=600)
        print(f"Saved all angular flux plot to {output_dir / f'all_angular_flux_{safe_label}.png'}")

    plt.close()


def plot_flux_difference(fortran_data, python_data, output_dir=None):
    """
    Plot the difference between Fortran and Python results.

    Parameters:
    -----------
    fortran_data : dict
        Fortran output data
    python_data : dict
        Python output data
    output_dir : Path or str, optional
        Directory to save plots
    """
    x_edges = np.array(fortran_data['mesh']['x_edges'])

    # Scalar flux difference
    scalar_flux_f = np.array(fortran_data['solution']['scalar_flux'])
    scalar_flux_p = np.array(python_data['solution']['scalar_flux'])
    scalar_diff = scalar_flux_f - scalar_flux_p

    # Current difference
    current_f = np.array(fortran_data['solution']['current'])
    current_p = np.array(python_data['solution']['current'])
    current_diff = current_f - current_p

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Scalar flux difference
    axes[0].plot(x_edges, scalar_diff, '-o', markersize=4, color='red')
    axes[0].set_xlabel('Position (cm)', fontsize=12)
    axes[0].set_ylabel('Difference', fontsize=12)
    axes[0].set_title('Scalar Flux Difference (Fortran - Python)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

    # Current difference
    axes[1].plot(x_edges, current_diff, '-o', markersize=4, color='blue')
    axes[1].set_xlabel('Position (cm)', fontsize=12)
    axes[1].set_ylabel('Difference', fontsize=12)
    axes[1].set_title('Current Difference (Fortran - Python)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'solution_differences.png', dpi=600)
        print(f"Saved difference plot to {output_dir / 'solution_differences.png'}")

    plt.close()

    # Print statistics
    print("\n" + "="*60)
    print("DIFFERENCE STATISTICS")
    print("="*60)
    print(f"Scalar Flux:")
    print(f"  Max absolute difference: {np.max(np.abs(scalar_diff)):.6e}")
    print(f"  Mean absolute difference: {np.mean(np.abs(scalar_diff)):.6e}")
    print(f"  RMS difference: {np.sqrt(np.mean(scalar_diff**2)):.6e}")
    print(f"\nCurrent:")
    print(f"  Max absolute difference: {np.max(np.abs(current_diff)):.6e}")
    print(f"  Mean absolute difference: {np.mean(np.abs(current_diff)):.6e}")
    print(f"  RMS difference: {np.sqrt(np.mean(current_diff**2)):.6e}")
    print("="*60 + "\n")


def main():
    # Use current working directory as the case directory
    case_dir = Path.cwd()

    print(f"Processing results from: {case_dir}")

    # Check for output files
    fortran_file = case_dir / "foutput.json"
    python_file = case_dir / "pyoutput.json"

    if not fortran_file.exists():
        print(f"Error: {fortran_file} not found!")
        print(f"Please run this script from a case directory containing output files.")
        return

    if not python_file.exists():
        print(f"Error: {python_file} not found!")
        print(f"Please run this script from a case directory containing output files.")
        return

    # Load data files
    print("Loading output files...")
    fortran_data = load_output_json(fortran_file)
    python_data = load_output_json(python_file)
    print("Data loaded successfully!\n")

    # Create output directory for plots
    output_dir = case_dir / "plots"

    # Prepare data dictionary for comparison plots
    data_dict = {
        'Fortran': fortran_data,
        'Python': python_data
    }

    # Plot scalar flux comparison
    print("Plotting scalar flux...")
    plot_scalar_flux(data_dict, output_dir)

    # Plot current comparison
    print("Plotting current...")
    plot_current(data_dict, output_dir)

    # Plot selected angular fluxes
    print("Plotting selected angular fluxes...")
    plot_angular_flux_selected_angles(data_dict, output_dir=output_dir)

    # Plot angular flux heatmaps
    print("Plotting angular flux heatmaps...")
    plot_angular_flux_heatmap(fortran_data, label="Fortran", output_dir=output_dir)
    plot_angular_flux_heatmap(python_data, label="Python", output_dir=output_dir)

    # Plot all angular fluxes with scalar flux envelope
    print("Plotting all angular fluxes with material regions...")
    plot_all_angular_fluxes(fortran_data, label="Fortran", output_dir=output_dir)
    plot_all_angular_fluxes(python_data, label="Python", output_dir=output_dir)

    # Plot differences
    print("Plotting differences...")
    plot_flux_difference(fortran_data, python_data, output_dir=output_dir)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()