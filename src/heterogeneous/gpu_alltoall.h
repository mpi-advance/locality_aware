#ifndef MPI_ADVANCE_GPU_ALLTOALL_H
#define MPI_ADVANCE_GPU_ALLTOALL_H

#include "collective/alltoall.h"
#include "collective/collective.h"

typedef int (*alltoall_ftn)(const void*, const int, MPI_Datatype, void*, const int, MPI_Datatype, MPI_Comm);

int gpu_aware_alltoall(alltoall_ftn f,
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int copy_to_cpu_alltoall(alltoall_ftn f,
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);

int gpu_aware_alltoall_pairwise(const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int gpu_aware_alltoall_nonblocking(const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int copy_to_cpu_alltoall_pairwise(const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int copy_to_cpu_alltoall_nonblocking(const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        MPIX_Comm* comm);

int threaded_alltoall_pairwise(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        MPIX_Comm* comm);

int threaded_alltoall_nonblocking(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        MPIX_Comm* comm);




/*
// TODO : how to share sendbuf and recvbuf (how to use duplicate device pointers??)
int ipc_copy_to_cpu_alltoall_alltoall(int (*f)(const void*, const int, MPI_Datatype, void*, const int, MPI_Dataype, MPI_Comm), 
        const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        const void* cpu_sendbuf,
        const void* gpu_sendbuf,
        MPIX_Comm* comm);

*/

#endif
