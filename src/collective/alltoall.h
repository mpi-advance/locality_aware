#ifndef MPI_ADVANCE_ALLTOALL_H
#define MPI_ADVANCE_ALLTOALL_H

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "utils.h"
#include "collective.h"
#include "locality/topology.h"
#include "persistent/persistent.h"

#ifdef __cplusplus
extern "C"
{
#endif

// Helper Functions
int alltoall_pairwise(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm);

int alltoall_bruck(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm);

// Locality-Aware Helper Functions
int alltoall_pairwise_loc(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);

int alltoall_rma(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);

int alltoall_rma_init(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr);

int alltoall_init(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr);

int alltoall_pairwise_init(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr);

int alltoall_nonblocking_init(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr);


#ifdef __cplusplus
}
#endif

#endif
