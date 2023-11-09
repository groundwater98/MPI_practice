from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()

shakespeare_config = {
    "batch_size": int(32/size),
    "epochs": 10,
    "buffer_size": int(10000/size)
}
cifar10_config = {
    "batch_size": int(32/size),
    "epochs": 10
}

dataset = "shakespeare"