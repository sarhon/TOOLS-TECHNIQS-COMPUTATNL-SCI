from mpi4py import MPI
comm = MPI.COMM_WORLD
# MPI.Init()
rank = comm.Get_rank()
size = comm.Get_size()
# print(f" {rank} {size}")

n_calc = 4

if rank == 0:
    # figures out what everything should do
    ranges = []
    for r in range(size):
        i_start = r * n_calc
        i_end = (r+1) * n_calc - 1
        ranges.append((i_start, i_end))

    for r in range(1, size):
        comm.send(
            ranges[r],  # data
            dest = r,   # destination
            tag=10      # tag
        )

        # what do send  | arg1
        # how much      | implied
        # type          | implied
        # dest          | arg2
        # tag           | arg3, for if you wanted to send multiple messages
        # comm          |

    my_range = ranges[0]

else:
    my_range = comm.recv(
        source = 0, # source
        tag = 10)   # tag


print(f" {rank} {my_range}")