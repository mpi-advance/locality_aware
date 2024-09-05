#include "collective/alltoall.h"
#include "collective/collective.h"
#include "gpu_alltoall.h"

// ASSUMES 1 CPU CORE PER GPU (Standard for applications)

int gpu_aware_alltoall(alltoall_ftn f,
        const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int ierr = f(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    return ierr;
}

int gpu_aware_alltoall_pairwise(const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return gpu_aware_alltoall(alltoall_pairwise,
        sendbuf, 
        sendcount,
        sendtype,
        recvbuf, 
        recvcount,
        recvtype,
        comm);
}

int gpu_aware_alltoall_nonblocking(const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return gpu_aware_alltoall(alltoall_nonblocking,
        sendbuf, 
        sendcount,
        sendtype,
        recvbuf, 
        recvcount,
        recvtype,
        comm);
}

int copy_to_cpu_alltoall(alltoall_ftn f,
        const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int ierr = 0;

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

    // Copy from GPU to CPU
    ierr += gpuMemcpy(cpu_sendbuf, sendbuf, total_bytes_s, gpuMemcpyDeviceToHost);

    // Collective Among CPUs
    ierr += f(cpu_sendbuf, sendcount, sendtype, cpu_recvbuf, recvcount, recvtype, comm);

    // Copy from CPU to GPU
    ierr += gpuMemcpy(recvbuf, cpu_recvbuf, total_bytes_r, gpuMemcpyHostToDevice);

    gpuFreeHost(cpu_sendbuf);
    gpuFreeHost(cpu_recvbuf);
    
    return ierr;
}

int copy_to_cpu_alltoall_pairwise(const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return copy_to_cpu_alltoall(alltoall_pairwise,
        sendbuf, 
        sendcount,
        sendtype,
        recvbuf, 
        recvcount,
        recvtype,
        comm);

}

int copy_to_cpu_alltoall_nonblocking(const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return copy_to_cpu_alltoall(alltoall_nonblocking,
        sendbuf, 
        sendcount,
        sendtype,
        recvbuf, 
        recvcount,
        recvtype,
        comm);
}

