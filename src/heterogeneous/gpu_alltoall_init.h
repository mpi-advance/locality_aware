#ifndef MPI_ADVANCE_GPU_ALLTOALL_H
#define MPI_ADVANCE_GPU_ALLTOALL_H

#include "collective/alltoall_init.h"
#include "collective/collective.h"

typedef int (*alltoall_init_ftn)(const void*, const int, MPI_Datatype, void*, const int, MPI_Datatype, MPI_Comm, MPI_Info, MPIX_Request**);

int gpu_aware_alltoall_init(alltoall_init_ftn f,
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr);
int gpu_aware_alltoall_init_nonblocking(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr);
int gpu_aware_alltoall_init_stride(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr);

/*
int copy_to_cpu_alltoall_init(alltoall_init_ftn f,
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Info info,
        MPIX_Request** request_ptr);
int copy_to_cpu_alltoall_init_nonblocking(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Info info,
        MPIX_Request** request_ptr);
int copy_to_cpu_alltoall_init_stride(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Info info,
        MPIX_Request** request_ptr);


int copy_neighbor_start(MPIX_Request* request);
int copy_neighbor_wait(MPIX_Request* request, MPI_Status status);



int threaded_neighbor_start(MPIX_Request* request);
int threaded_neighbor_wait(MPIX_Request* request, MPI_Status status);
*/

#endif
