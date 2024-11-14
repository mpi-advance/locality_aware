#ifndef MPI_ADVANCE_ALLTOALL_INIT_H
#define MPI_ADVANCE_ALLTOALL_INIT_H

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "utils/utils.h"
#include "collective.h"
#include "locality/topology.hpp"
#include "persistent/persistent.h"
#include "neighborhood/neighbor_persistent.h"

#ifdef __cplusplus
extern "C"
{
#endif

// Helper Functions
int alltoall_init_nonblocking(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Info info,
        MPIX_Request** request_ptr);
int alltoall_init_stride(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Info info,
        MPIX_Request** request_ptr);
int alltoall_init_nonblocking_helper(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Info info,
        MPIX_Request** request_ptr);

int partial_neighbor_start(MPIX_Request* request);
int partial_neighbor_wait(MPIX_Request* request, MPI_Status status);


#ifdef __cplusplus
}
#endif

#endif
