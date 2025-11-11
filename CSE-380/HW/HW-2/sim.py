import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpi4py import MPI

# ============================================================================
# PERFORMANCE TOGGLE: Enable/disable Numba JIT compilation
# Read from environment variable USE_NUMBA (default: False if not set)
# Usage: USE_NUMBA=0 mpirun -np 8 python main.py  (disabled)
#        USE_NUMBA=1 mpirun -np 8 python main.py  (enabled)
#        mpirun -np 8 python main.py              (disabled by default)
# ============================================================================
# Check environment variable (accepts: "1", "true", "True", "yes", "Yes")
USE_NUMBA_ENV = os.getenv('USE_NUMBA', '').lower()
USE_NUMBA = USE_NUMBA_ENV in ('1', 'true', 'yes')

if USE_NUMBA:
    try:
        from numba import jit, prange
    except ImportError:
        print("Warning: Numba not available. Falling back to pure Python.")
        USE_NUMBA = False

# Print which mode is active
comm_init = MPI.COMM_WORLD
if comm_init.Get_rank() == 0:
    print(f"Running with Numba optimization: {USE_NUMBA}\n")



def init_cluster_with_core(N=1000, G=1.0, Mc=1000.0, Rmax=1.0, seed=42):
    """
    Initialize a globular cluster with N stars orbiting a central mass Mc.

    Parameters:
    -----------
    N : int
        Number of stars
    G : float
        Gravitational constant
    Mc : float
        Central mass (equals total stellar mass)
    Rmax : float
        Maximum radius of initial sphere
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    pos : ndarray (N, 3)
        Initial positions
    vel : ndarray (N, 3)
        Initial velocities
    mass : ndarray (N,)
        Mass of each star
    Mc : float
        Central mass
    """
    rng = np.random.default_rng(seed)

    # Positions inside sphere (uniformly distributed in 3D)
    u = rng.normal(size=(N, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    R = rng.random(N) ** (1 / 3) * Rmax
    pos = u * R[:, None]

    # Speeds: v(r) = sqrt(1.5*G*Mc / r)
    r = np.maximum(np.linalg.norm(pos, axis=1), 1e-6)
    vmag = np.sqrt(1.0 * G * Mc / r)

    # Random velocity directions (isotropic)
    w = rng.normal(size=(N, 3))
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    vel = w * vmag[:, None]

    # Equal mass stars
    mass = np.full(N, Mc / N)

    return pos, vel, mass, Mc


# ============================================================================
# Numba-optimized kernels (only compiled if USE_NUMBA=True)
# ============================================================================
if USE_NUMBA:
    @jit(nopython=True, parallel=True, fastmath=True)
    def compute_pairwise_accel(pos, mass, accel_local, start_idx, end_idx, G, eps2):
        """
        Numba-compiled pairwise acceleration kernel.

        Key optimizations:
        - nopython=True: Full compilation to machine code
        - parallel=True: Automatic parallelization with OpenMP
        - fastmath=True: Relaxed floating-point constraints for speed
        """
        N = pos.shape[0]

        # Parallel loop over particles (uses OpenMP threads)
        for i in prange(start_idx, end_idx):
            local_i = i - start_idx
            ax, ay, az = 0.0, 0.0, 0.0

            # Inner loop over all interactions
            for j in range(N):
                if i == j:
                    continue

                # Compute separation vector
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]

                # Compute distance
                dist_sq = dx * dx + dy * dy + dz * dz + eps2
                dist = np.sqrt(dist_sq)
                dist_inv3 = 1.0 / (dist * dist * dist)

                # Accumulate acceleration
                factor = G * mass[j] * dist_inv3
                ax += factor * dx
                ay += factor * dy
                az += factor * dz

            accel_local[local_i, 0] = ax
            accel_local[local_i, 1] = ay
            accel_local[local_i, 2] = az
else:
    # NumPy-vectorized fallback (no Numba, but still optimized)
    def compute_pairwise_accel(pos, mass, accel_local, start_idx, end_idx, G, eps2):
        """
        NumPy-vectorized pairwise acceleration kernel (fallback when Numba unavailable).
        Uses broadcasting to vectorize the inner loop for better performance.
        """
        for local_i, i in enumerate(range(start_idx, end_idx)):
            # Vector from star i to all other stars (broadcasting)
            r_ij = pos - pos[i]  # Shape: (N, 3)

            # Distance from star i to all others
            dist = np.sqrt(np.sum(r_ij ** 2, axis=1) + eps2)

            # Avoid self-interaction by setting distance to infinity
            dist[i] = np.inf

            # Acceleration contribution from all other stars (vectorized)
            accel_local[local_i] = np.sum(G * mass[:, None] * r_ij / dist[:, None] ** 3, axis=0)

def compute_acceleration(pos, mass, Mc, G=1.0, eps2=1e-4):
    """
    Compute gravitational acceleration on each star (MPI-parallelized).

    Parameters:
    -----------
    pos : ndarray (N, 3)
        Current positions
    mass : ndarray (N,)
        Mass of each star
    Mc : float
        Central mass
    G : float
        Gravitational constant
    eps2 : float
        Softening parameter (squared) to avoid singularities

    Returns:
    --------
    accel : ndarray (N, 3)
        Acceleration on each star
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = len(mass)
    accel = np.zeros_like(pos)

    # 1. Acceleration from central mass (each rank computes full array)
    r_center = pos  # Position vectors from origin
    r_mag = np.sqrt(np.sum(r_center ** 2, axis=1) + eps2)
    accel_center = -G * Mc * r_center / r_mag[:, None] ** 3
    accel += accel_center

    # 2. Acceleration from pairwise star interactions (parallelized)
    # Divide particles among ranks
    particles_per_rank = N // size
    start_idx = rank * particles_per_rank
    if rank == size - 1:
        end_idx = N  # Last rank handles remainder
    else:
        end_idx = (rank + 1) * particles_per_rank

    # Each rank computes acceleration for its subset
    accel_local = np.zeros((end_idx - start_idx, 3))

    compute_pairwise_accel(pos, mass, accel_local, start_idx, end_idx, G, eps2)

    # for local_i, i in enumerate(range(start_idx, end_idx)):
    #     # Vector from star i to all other stars
    #     r_ij = pos - pos[i]  # Shape: (N, 3)
    #
    #     # Distance from star i to all others
    #     dist = np.sqrt(np.sum(r_ij ** 2, axis=1) + eps2)
    #
    #     # Avoid self-interaction by setting distance to infinity
    #     dist[i] = np.inf
    #
    #     # Acceleration contribution from all other stars
    #     accel_local[local_i] += np.sum(G * mass[:, None] * r_ij / dist[:, None] ** 3, axis=0)

    # Gather results from all ranks
    # Prepare receive buffer and counts
    counts = [particles_per_rank * 3] * size
    counts[-1] = (N - (size - 1) * particles_per_rank) * 3  # Last rank might have more

    accel_pairwise = np.zeros((N, 3))
    comm.Allgatherv(accel_local.flatten(), [accel_pairwise.flatten(), counts])

    accel += accel_pairwise

    return accel


def velocity_verlet_step(pos, vel, mass, Mc, dt, accel_old, G=1.0, eps2=1e-4):
    """
    Perform one velocity-Verlet integration step (efficient version).

    This version reuses the acceleration from the previous step to avoid
    redundant computation, improving performance by ~2x.

    Parameters:
    -----------
    pos : ndarray (N, 3)
        Current positions
    vel : ndarray (N, 3)
        Current velocities
    mass : ndarray (N,)
        Mass of each star
    Mc : float
        Central mass
    dt : float
        Time step
    accel_old : ndarray (N, 3)
        Acceleration at current positions (from previous step)
    G : float
        Gravitational constant
    eps2 : float
        Softening parameter

    Returns:
    --------
    pos_new : ndarray (N, 3)
        Updated positions
    vel_new : ndarray (N, 3)
        Updated velocities
    accel_new : ndarray (N, 3)
        Acceleration at new positions (for next step)
    """
    # Update positions: r_new = r + v*dt + 0.5*a*dt^2
    pos_new = pos + vel * dt + 0.5 * accel_old * dt ** 2

    # Compute acceleration at new positions
    accel_new = compute_acceleration(pos_new, mass, Mc, G, eps2)

    # Update velocities: v_new = v + 0.5*(a_old + a_new)*dt
    vel_new = vel + 0.5 * (accel_old + accel_new) * dt

    return pos_new, vel_new, accel_new


def run_simulation(N=1000, dt=0.001, n_steps=1000, snapshot_interval=10):
    """
    Run the globular cluster simulation.

    Parameters:
    -----------
    N : int
        Number of stars
    dt : float
        Time step
    n_steps : int
        Number of integration steps
    snapshot_interval : int
        Save positions every this many steps

    Returns:
    --------
    snapshots : list of ndarrays
        Position snapshots for visualization
    vel_snapshots : list of ndarrays
        Velocity snapshots for energy analysis
    times : ndarray
        Times corresponding to snapshots
    mass : ndarray
        Mass of each star
    Mc : float
        Central mass
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialize the cluster
    if rank == 0:
        print(f"Initializing cluster with N={N} stars...\n")
    pos, vel, mass, Mc = init_cluster_with_core(N=N)

    # Storage for snapshots
    snapshots = []
    vel_snapshots = []
    times = []

    # Save initial state
    snapshots.append(pos.copy())
    vel_snapshots.append(vel.copy())
    times.append(0.0)

    if rank == 0:
        print(f"Running simulation for {n_steps:.1E} steps (dt={dt:.1E})...")

    # Compute initial acceleration (for efficient velocity Verlet)
    accel = compute_acceleration(pos, mass, Mc)

    # Main integration loop (efficient: reuses acceleration from previous step)
    for step in range(n_steps):
        # Perform one time step - returns new acceleration for next iteration
        pos, vel, accel = velocity_verlet_step(pos, vel, mass, Mc, dt, accel)

        # Save snapshot
        if (step + 1) % snapshot_interval == 0:
            snapshots.append(pos.copy())
            vel_snapshots.append(vel.copy())
            times.append((step + 1) * dt)

        # Progress update (only rank 0 prints)
        if rank == 0 and (step + 1) % round(n_steps*1e-2) == 0:
            print(f"    {(step + 1)/(n_steps)*1e2:.2F}% completed", end='\r')

    if rank == 0:
        print()  # Newline after progress
        print("Simulation complete!")

    return snapshots, vel_snapshots, np.array(times), mass, Mc


def visualize_snapshot(pos, title="Globular Cluster"):
    """
    Create a simple 2D visualization of the cluster.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot stars in x-y plane
    ax.scatter(pos[:, 0], pos[:, 1], s=1, color='white', alpha=0.6)

    # Plot central mass as a larger point
    ax.scatter([0], [0], s=100, color='yellow', marker='*',
               edgecolors='orange', linewidth=1, label='Central Mass')

    ax.set_facecolor('black')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, color='white')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    return fig, ax


def create_2d_animation(snapshots, times, output_file='cluster_2d.mp4',
                        fps=30, dpi=150, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5)):
    """
    Create a 2D animated GIF of the cluster evolution (x-y plane).

    Parameters:
    -----------
    snapshots : list of ndarrays
        Position snapshots from simulation
    times : ndarray
        Time values for each snapshot
    output_file : str
        Output filename for the GIF
    fps : int
        Frames per second
    dpi : int
        Resolution (dots per inch)
    xlim : tuple
        Fixed x-axis limits (min, max)
    ylim : tuple
        Fixed y-axis limits (min, max)
    """
    print(f"Creating 2D animation with {len(snapshots)} frames...")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Set fixed plot limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel('x', fontsize=12, color='white')
    ax.set_ylabel('y', fontsize=12, color='white')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='white')

    # Initialize scatter plots
    scat_stars = ax.scatter([], [], s=2, color='white', alpha=0.6, label='Stars')
    scat_center = ax.scatter([0], [0], s=150, color='yellow', marker='*',
                             edgecolors='orange', linewidth=2, label='Central Mass',
                             zorder=5)

    title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                    ha='center', fontsize=14, color='white')

    ax.legend(loc='upper right', facecolor='black', edgecolor='white',
              labelcolor='white')

    def update(frame):
        """Update function for animation."""
        pos = snapshots[frame]
        scat_stars.set_offsets(pos[:, :2])  # x-y projection
        title.set_text(f'Globular Cluster: t = {times[frame]:.3f}')
        return scat_stars, scat_center, title

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(snapshots),
                         interval=1000 / fps, blit=True)

    # Save as MP4 (much faster than GIF)
    writer = FFMpegWriter(fps=fps, bitrate=1800, codec='libx264')
    anim.save(output_file, writer=writer, dpi=dpi)
    print(f"Saved 2D animation to {output_file}")
    plt.close(fig)

    return output_file


if USE_NUMBA:
    @jit(nopython=True, parallel=True, fastmath=True)
    def compute_pairwise_energy(pos, mass, start_idx, end_idx, G, eps2):
        """
        Numba-compiled pairwise potential energy kernel.
        """
        N = pos.shape[0]
        potential = 0.0

        # Parallel reduction over particles
        for i in prange(start_idx, end_idx):
            local_potential = 0.0

            for j in range(N):
                if i == j:
                    continue

                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]

                dist = np.sqrt(dx * dx + dy * dy + dz * dz + eps2)
                local_potential += G * mass[i] * mass[j] / dist

            potential += local_potential

        return potential


    @jit(nopython=True, fastmath=True)
    def compute_radii(pos):
        """
        Numba-compiled function to compute radial distances.

        For small N, this might not be faster than np.linalg.norm,
        but it avoids temporary arrays and can be faster for large N.
        """
        N = pos.shape[0]
        radii = np.empty(N)
        for i in range(N):
            radii[i] = np.sqrt(pos[i, 0]**2 + pos[i, 1]**2 + pos[i, 2]**2)
        return radii
else:
    # NumPy-vectorized fallback (no Numba, but still optimized)
    def compute_pairwise_energy(pos, mass, start_idx, end_idx, G, eps2):
        """
        NumPy-vectorized pairwise potential energy kernel (fallback when Numba unavailable).
        Uses broadcasting to vectorize the inner loop for better performance.
        """
        potential = 0.0

        for i in range(start_idx, end_idx):
            # Vector from star i to all other stars (broadcasting)
            r_ij = pos - pos[i]  # Shape: (N, 3)

            # Distance from star i to all others
            dist = np.sqrt(np.sum(r_ij ** 2, axis=1) + eps2)

            # Avoid self-interaction by setting distance to infinity
            dist[i] = np.inf

            # Potential energy contribution from all other stars (vectorized)
            potential += np.sum(G * mass[i] * mass / dist)

        return potential


    def compute_radii(pos):
        """
        NumPy-vectorized function to compute radial distances (fallback when Numba unavailable).
        This is actually the optimal implementation - faster than Numba for small-medium N.
        """
        return np.linalg.norm(pos, axis=1)

def compute_energy(pos, vel, mass, Mc=1000.0, G=1.0, eps2=1e-4):
    """
    Compute total energy of the system (MPI-parallelized).

    Parameters:
    -----------
    pos : ndarray (N, 3)
        Current positions
    vel : ndarray (N, 3)
        Current velocities
    mass : ndarray (N,)
        Mass of each star
    Mc : float
        Central mass
    G : float
        Gravitational constant
    eps2 : float
        Softening parameter (squared)

    Returns:
    --------
    total_energy : float
        Total energy (kinetic + potential)
    mean_energy : float
        Mean energy per unit mass
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = len(mass)

    # Kinetic energy (computed by all ranks, same result)
    kinetic = 0.5 * np.sum(mass * np.sum(vel ** 2, axis=1))

    # Potential energy from pairwise interactions (parallelized with Numba)
    # Divide particles among ranks
    particles_per_rank = N // size
    start_idx = rank * particles_per_rank
    if rank == size - 1:
        end_idx = N  # Last rank handles remainder
    else:
        end_idx = (rank + 1) * particles_per_rank

    # Each rank computes potential energy for its subset using Numba
    potential_local = compute_pairwise_energy(pos, mass, start_idx, end_idx, G, eps2)

    # Sum partial potentials from all ranks
    potential_pairwise = comm.allreduce(potential_local, op=MPI.SUM)
    potential_pairwise *= 0.5  # Avoid double-counting pairs

    # Central potential contribution (computed by all ranks, same result)
    r_core = np.sqrt(np.sum(pos ** 2, axis=1) + eps2)
    potential_center = np.sum(G * Mc * mass / r_core)

    # Total potential and energy
    potential = potential_pairwise + potential_center
    total_energy = kinetic + potential
    mean_energy = total_energy / np.sum(mass)

    return total_energy, mean_energy


def plot_energy_conservation(snapshots, vel_snapshots, times, mass, Mc,
                             G=1.0, eps2=1e-4, output_file='energy_conservation.png'):
    """
    Plot total energy vs time to verify energy conservation.

    Parameters:
    -----------
    snapshots : list of ndarrays
        Position snapshots from simulation
    vel_snapshots : list of ndarrays
        Velocity snapshots from simulation
    times : ndarray
        Time values for each snapshot
    mass : ndarray
        Mass of each star
    Mc : float
        Central mass
    G : float
        Gravitational constant
    eps2 : float
        Softening parameter
    output_file : str
        Output filename for the plot

    Returns:
    --------
    fig, ax : matplotlib figure and axes
        The created plot objects
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Computing energy at each snapshot (MPI-parallelized)...")

    energies = []
    n_snapshots = len(snapshots)
    for i, (pos, vel) in enumerate(zip(snapshots, vel_snapshots)):
        total_energy, _ = compute_energy(pos, vel, mass, Mc, G, eps2)
        energies.append(total_energy)
        if rank == 0 and (i + 1) % round(n_steps*1e-2) == 0:
            print(f"    {(i + 1)/(n_snapshots)*1e2:.2F}% completed", end='\r')

    if rank == 0:
        print()  # Newline after progress
        energies = np.array(energies)

        # Calculate energy drift
        E0 = energies[0]
        energy_drift = (energies - E0) / np.abs(E0) * 100  # Percentage drift

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot 1: Absolute energy
        ax1.plot(times, energies, 'b-', linewidth=1.5)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Total Energy', fontsize=12)
        ax1.set_title('Energy Conservation: Total Energy vs Time', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=E0, color='r', linestyle='--', linewidth=1,
                    label=f'Initial Energy: {E0:.4f}')
        ax1.legend()

        # Plot 2: Relative energy drift (%)
        ax2.plot(times, energy_drift, 'r-', linewidth=1.5)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Energy Drift (%)', fontsize=12)
        ax2.set_title('Relative Energy Drift', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

        # Add statistics text
        final_drift = energy_drift[-1]
        max_drift = np.max(np.abs(energy_drift))
        stats_text = f'Final drift: {final_drift:.4f}%\nMax drift: {max_drift:.4f}%'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round',
                                                    facecolor='wheat', alpha=0.5), fontsize=10)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')

        print(f"Saved energy conservation plot to {output_file}")

        print(f"""
Energy Conservation Summary:
    Initial Energy: {E0:.6f}
    Final Energy:   {energies[-1]:.6f}
    Final Drift:    {final_drift:.4f}%
    Max Drift:      {max_drift:.4f}%""")

        return fig, (ax1, ax2)
    return


def plot_median_radius(snapshots, times, output_file='median_radius.png'):
    """
    Plot median radius from center vs time to check cluster stability.
    MPI-parallelized for faster computation.

    Parameters:
    -----------
    snapshots : list of ndarrays
        Position snapshots from simulation
    times : ndarray
        Time values for each snapshot
    output_file : str
        Output filename for the plot

    Returns:
    --------
    fig, ax : matplotlib figure and axes
        The created plot objects
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("Computing median radius at each snapshot (MPI-parallelized)...")

    n_snapshots = len(snapshots)

    # Divide snapshots among MPI ranks
    snapshots_per_rank = n_snapshots // size
    start_idx = rank * snapshots_per_rank
    if rank == size - 1:
        end_idx = n_snapshots  # Last rank handles remainder
    else:
        end_idx = (rank + 1) * snapshots_per_rank

    # Each rank computes median radii for its subset of snapshots
    local_median_radii = []
    local_count = 0
    for i in range(start_idx, end_idx):
        pos = snapshots[i]
        # # Option 1: Use NumPy (usually faster for small-medium N)
        # radii = np.linalg.norm(pos, axis=1)
        # Option 2: Use Numba (can be faster for large N or many snapshots)
        radii = compute_radii(pos)

        local_median_radii.append(np.median(radii))
        local_count += 1

        # Progress printing: each rank reports its own progress
        if rank == 0 and (i + 1) % round(n_steps*1e-2) == 0:
            print(f"    {(i + 1)/(n_snapshots)*1e2:.2F}% completed", end='\r')

    # Gather all median radii to rank 0
    all_median_radii = comm.gather(local_median_radii, root=0)

    if rank == 0:
        print()  # Newline after progress
        # Flatten the gathered list and convert to array
        median_radii = np.array([item for sublist in all_median_radii for item in sublist])

        # Calculate radius change
        r0 = median_radii[0]
        radius_change = (median_radii - r0) / r0 * 100  # Percentage change

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot 1: Absolute median radius
        ax1.plot(times, median_radii, 'g-', linewidth=1.5)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Median Radius', fontsize=12)
        ax1.set_title('Cluster Stability: Median Radius vs Time', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=r0, color='r', linestyle='--', linewidth=1,
                    label=f'Initial Median Radius: {r0:.4f}')
        ax1.legend()

        # Plot 2: Relative radius change (%)
        ax2.plot(times, radius_change, 'm-', linewidth=1.5)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Radius Change (%)', fontsize=12)
        ax2.set_title('Relative Radius Change', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

        # Add statistics text
        final_change = radius_change[-1]
        max_change = np.max(np.abs(radius_change))
        stats_text = f'Final change: {final_change:.4f}%\nMax change: {max_change:.4f}%'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round',
                                                    facecolor='lightblue', alpha=0.5), fontsize=10)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved median radius plot to {output_file}")

        print(f"""
Median Radius Summary:
    Initial Radius: {r0:.6f}
    Final Radius:   {median_radii[-1]:.6f}
    Final Change:   {final_change:.4f}%
    Max Change:     {max_change:.4f}%""")

        return fig, (ax1, ax2)

    return None


def create_3d_animation(snapshots, times, output_file='cluster_3d.mp4',
                        fps=30, dpi=150, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
                        zlim=(-1.5, 1.5), elev=20, azim=45):
    """
    Create a 3D animated GIF of the cluster evolution.

    Parameters:
    -----------
    snapshots : list of ndarrays
        Position snapshots from simulation
    times : ndarray
        Time values for each snapshot
    output_file : str
        Output filename for the GIF
    fps : int
        Frames per second
    dpi : int
        Resolution (dots per inch)
    xlim : tuple
        Fixed x-axis limits (min, max)
    ylim : tuple
        Fixed y-axis limits (min, max)
    zlim : tuple
        Fixed z-axis limits (min, max)
    elev : float
        Elevation angle for 3D view (degrees)
    azim : float
        Azimuthal angle for 3D view (degrees)
    """
    print(f"Creating 3D animation with {len(snapshots)} frames...")

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Make grid lines more visible
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(True, alpha=0.3, color='white')

    # Set fixed plot limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_xlabel('x', fontsize=10, color='white')
    ax.set_ylabel('y', fontsize=10, color='white')
    ax.set_zlabel('z', fontsize=10, color='white')

    # Set tick colors
    ax.tick_params(colors='white', labelsize=8)

    # Initialize scatter plot
    scat_stars = ax.scatter([], [], [], s=3, color='white', alpha=0.6,
                            label='Stars')
    scat_center = ax.scatter([0], [0], [0], s=200, color='yellow', marker='*',
                             edgecolors='orange', linewidth=2, label='Central Mass',
                             depthshade=False)

    title = ax.text2D(0.5, 0.95, '', transform=ax.transAxes,
                      ha='center', fontsize=14, color='white')

    ax.legend(loc='upper right', facecolor='black', edgecolor='white',
              labelcolor='white')

    # Set fixed viewing angle
    ax.view_init(elev=elev, azim=azim)

    def update(frame):
        """Update function for animation."""
        pos = snapshots[frame]
        scat_stars._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        title.set_text(f'Globular Cluster (3D): t = {times[frame]:.3f}')

        return scat_stars, scat_center, title

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(snapshots),
                         interval=1000 / fps, blit=False)

    # Save as MP4 (much faster than GIF)
    writer = FFMpegWriter(fps=fps, bitrate=1800, codec='libx264')
    anim.save(output_file, writer=writer, dpi=dpi)
    print(f"Saved 3D animation to {output_file}")
    plt.close(fig)

    return output_file


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Run a test simulation with modest parameters
    # N = round(64*8)  # Number of stars
    N = round(64*8)  # Number of stars
    resolution = 1e-6
    dt = resolution
    n_steps = round(resolution ** -1)
    snapshot_intervals = int(max([1e-3 * n_steps, 1]))

    # Run the simulation (all ranks participate)
    run = True
    make_animations = False

    if run:
        snapshots, vel_snapshots, times, mass, Mc = run_simulation(N=N, dt=dt, n_steps=n_steps,
                                                                   snapshot_interval=snapshot_intervals)

        # Only rank 0 saves data
        if rank == 0:
            data = {
                'snapshots': snapshots,
                'vel_snapshots': vel_snapshots,
                'times': times,
                'mass': mass,
                'Mc': Mc
            }

            print('Saving Data')
            with open('./data.pkl', 'wb') as f:
                pickle.dump(data, f)

        # Wait for rank 0 to finish saving before all ranks try to load
        comm.Barrier()

    # Load data on all ranks (needed for MPI-parallel energy computation)
    if rank == 0:
        print('Loading Data')

    with open('./data.pkl', 'rb') as f:
        data = pickle.load(f)

    snapshots = data['snapshots']
    vel_snapshots = data['vel_snapshots']
    times = data['times']
    mass = data['mass']
    Mc = data['Mc']

    # Only rank 0 does visualization
    if rank == 0:
        # Visualize initial and final states
        fig1, ax1 = visualize_snapshot(snapshots[0], title=f"Initial State (t=0)")
        plt.savefig('./initial_state.png', dpi=150, facecolor='black')
        print(f"Saved initial state visualization")

        fig2, ax2 = visualize_snapshot(snapshots[-1],
                                       title=f"Final State (t={times[-1]:.3f})")
        plt.savefig('./final_state.png', dpi=150, facecolor='black')
        print(f"Saved final state visualization")

        print("\n" + "=" * 50)
        print("Plotting energy conservation...")
        print("=" * 50)

    # Energy conservation (all ranks participate for MPI-parallel computation)
    plot_energy_conservation(snapshots, vel_snapshots, times, mass, Mc,
                         output_file='./energy_conservation.png')

    # Median radius (all ranks participate for MPI-parallel computation)
    if rank == 0:
        print("\n" + "=" * 50)
        print("Plotting median radius...")
        print("=" * 50)
    plot_median_radius(snapshots, times,
                       output_file='./median_radius.png')

    # Only rank 0 does animations
    if rank == 0 and make_animations:
        print("\n" + "=" * 50)
        print("Creating animations...")
        print("=" * 50)

        # 2D animation with fixed boundaries
        create_2d_animation(snapshots, times,
                            output_file='./cluster_2d.mp4',
                            fps=30, dpi=150,
                            xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))

        # 3D animation with fixed boundaries and no rotation
        create_3d_animation(snapshots, times,
                            output_file='./cluster_3d.mp4',
                            fps=30, dpi=150,
                            xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5),
                            elev=20, azim=45)

    if rank == 0:
        print("\n" + "=" * 50)
        print("All visualizations complete!")
        print("=" * 50)
