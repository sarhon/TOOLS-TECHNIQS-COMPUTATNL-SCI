import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Total number of Fibonacci numbers to compute
total_n = 128*8

# Fibonacci matrix: [[1, 1], [1, 0]]
# Property: [[F(n+1), F(n)], [F(n), F(n-1)]] = M^n
M = np.array([[1, 1],
              [1, 0]], dtype=object)

def fib_matrix(n):
    """
    Compute F(n) using matrix exponentiation.
    [[F(n+1), F(n)], [F(n), F(n-1)]] = [[1, 1], [1, 0]]^n
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1

    # For n >= 2, use matrix exponentiation
    M_n = np.linalg.matrix_power(M, n)
    return int(M_n[0, 1])  # F(n) is at position [0, 1]

# Distribute work among processes
n_per_rank = total_n // size
remainder = total_n % size

# Calculate start and end indices for this rank
if rank < remainder:
    # First 'remainder' ranks get one extra number
    start_idx = rank * (n_per_rank + 1)
    end_idx = start_idx + n_per_rank + 1
else:
    start_idx = remainder * (n_per_rank + 1) + (rank - remainder) * n_per_rank
    end_idx = start_idx + n_per_rank

# Compute Fibonacci numbers for this rank's range
local_fibs = []
for i in range(start_idx, end_idx):
    fib_val = fib_matrix(i)
    local_fibs.append((i, fib_val))

if rank == 0:
    print(f"Computing Fibonacci numbers 0 to {total_n-1} using {size} MPI processes")
    print(f"Using matrix formulation: [[F(n+1), F(n)], [F(n), F(n-1)]] = [[1,1],[1,0]]^n")
    print("-" * 80)

# Gather all results to rank 0
all_fibs = comm.gather(local_fibs, root=0)

if rank == 0:
    # Flatten the list of lists and sort by index
    all_results = []
    for local_list in all_fibs:
        all_results.extend(local_list)
    all_results.sort(key=lambda x: x[0])

    # Display first 20 and last 10
    print("\nFirst 20 Fibonacci numbers:")
    for i, fib in all_results[:20]:
        print(f"F({i:3d}) = {fib}")

    print("\n...")
    print("\nLast 10 Fibonacci numbers:")
    for i, fib in all_results[-10:]:
        print(f"F({i:3d}) = {fib}")

    print(f"\n{'-' * 80}")
    print(f"Total computed: {len(all_results)} Fibonacci numbers")
    print(f"Largest: F({total_n-1}) has {len(str(all_results[-1][1]))} digits")