from mpi4py import MPI

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
rank_size = comm.Get_size()

print("myrank is ", myrank)
print("rank size is ", rank_size)
