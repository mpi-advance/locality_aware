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
    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int ierr, mpi_err;
    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    //int total_bytes_s = sendcount * send_bytes * num_procs;
    //int total_bytes_r = recvcount * recv_bytes * num_procs;
    int total_bytes_s = sendcount * send_bytes;
    int total_bytes_r = recvcount * recv_bytes;

    char* cpu_sendbuf;
    char* cpu_recvbuf;
    ierr = gpuMallocHost((void**)&cpu_sendbuf, total_bytes_s);
    gpu_check(ierr);
    ierr = gpuMallocHost((void**)&cpu_recvbuf, total_bytes_r);
    gpu_check(ierr);

    mpi_err = f(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

    ierr = gpuFreeHost(cpu_sendbuf);
    gpu_check(ierr);
    ierr = gpuFreeHost(cpu_recvbuf);
    gpu_check(ierr);

    return mpi_err;
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
    int ierr;
    int mpi_err = 0;

    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int total_bytes_s = sendcount * send_bytes * num_procs;
    int total_bytes_r = recvcount * recv_bytes * num_procs;

    char* cpu_sendbuf;
    char* cpu_recvbuf;
    ierr = gpuMallocHost((void**)&cpu_sendbuf, total_bytes_s);
    gpu_check(ierr);
    ierr = gpuMallocHost((void**)&cpu_recvbuf, total_bytes_r);
    gpu_check(ierr);

    // Copy from GPU to CPU
    mpi_err += gpuMemcpy(cpu_sendbuf, sendbuf, total_bytes_s, gpuMemcpyDeviceToHost);

    // Collective Among CPUs
    mpi_err += f(cpu_sendbuf, sendcount, sendtype, cpu_recvbuf, recvcount, recvtype, comm);

    // Copy from CPU to GPU
    mpi_err += gpuMemcpy(recvbuf, cpu_recvbuf, total_bytes_r, gpuMemcpyHostToDevice);

    ierr = gpuFreeHost(cpu_sendbuf);
    gpu_check(ierr);
    ierr = gpuFreeHost(cpu_recvbuf);
    gpu_check(ierr);
    
    return mpi_err;
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

