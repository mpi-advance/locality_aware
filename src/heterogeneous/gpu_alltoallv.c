#include "collective/alltoallv.h"
#include "collective/collective.h"
#include "gpu_alltoallv.h"

// ASSUMES 1 CPU CORE PER GPU (Standard for applications)

int gpu_aware_alltoallv(alltoallv_ftn f,
        const void* sendbuf, 
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return f(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
            comm->global_comm);
}

int gpu_aware_alltoallv_pairwise(const void* sendbuf, 
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return gpu_aware_alltoallv(alltoallv_pairwise,
        sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf,
        recvcounts,
        rdispls,
        recvtype,
        comm);
}

int gpu_aware_alltoallv_nonblocking(const void* sendbuf, 
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return gpu_aware_alltoallv(alltoallv_nonblocking,
        sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf, 
        recvcounts,
        rdispls,
        recvtype,
        comm);
}

int copy_to_cpu_alltoallv(alltoallv_ftn f,
        const void* sendbuf, 
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int ierr = 0;

    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int sendcount = 0;
    int recvcount = 0;
    for (int i = 0; i < num_procs; i++)
    {
        sendcount += sendcounts[i];
        recvcount += recvcounts[i];
    }

    int total_bytes_s = sendcount * send_bytes;
    int total_bytes_r = recvcount * recv_bytes;

    char* cpu_sendbuf;
    char* cpu_recvbuf;
    gpuMallocHost((void**)&cpu_sendbuf, total_bytes_s);
    gpuMallocHost((void**)&cpu_recvbuf, total_bytes_r);

    // Copy from GPU to CPU
    ierr += gpuMemcpy(cpu_sendbuf, sendbuf, total_bytes_s, gpuMemcpyDeviceToHost);

    // Collective Among CPUs
    ierr += f(cpu_sendbuf, sendcounts, sdispls, sendtype, 
            cpu_recvbuf, recvcounts, rdispls, recvtype, comm->global_comm);

    // Copy from CPU to GPU
    ierr += gpuMemcpy(recvbuf, cpu_recvbuf, total_bytes_r, gpuMemcpyHostToDevice);

    gpuFreeHost(cpu_sendbuf);
    gpuFreeHost(cpu_recvbuf);
    
    return ierr;
}

int copy_to_cpu_alltoallv_pairwise(const void* sendbuf, 
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return copy_to_cpu_alltoallv(alltoallv_pairwise,
        sendbuf, 
        sendcounts,
        sdispls,
        sendtype,
        recvbuf, 
        recvcounts,
        rdispls,
        recvtype,
        comm);

}

int copy_to_cpu_alltoallv_nonblocking(const void* sendbuf, 
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return copy_to_cpu_alltoallv(alltoallv_nonblocking,
        sendbuf, 
        sendcounts,
        sdispls,
        sendtype,
        recvbuf, 
        recvcounts,
        rdispls,
        recvtype,
        comm);
}

