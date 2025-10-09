#include "../../include/heterogeneous/gpu_alltoall.h"

#include "../../include/collective/alltoall.h"
#include "../../../include/collective/collective.h"

// ASSUMES 1 CPU CORE PER GPU (Standard for applications)
int gpu_aware_alltoall(alltoall_ftn f,
                       const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm)
{
    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int total_bytes_s = sendcount * send_bytes * num_procs;
    int total_bytes_r = recvcount * recv_bytes * num_procs;

    char* cpu_sendbuf;
    char* cpu_recvbuf;
    gpuMallocHost((void**)&cpu_sendbuf, total_bytes_s);
    gpuMallocHost((void**)&cpu_recvbuf, total_bytes_r);

    int ierr = f(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

    gpuFreeHost(cpu_sendbuf);
    gpuFreeHost(cpu_recvbuf);

    return ierr;
}