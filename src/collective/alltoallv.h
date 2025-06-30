#ifndef MPI_ADVANCE_ALLTOALLV_H
#define MPI_ADVANCE_ALLTOALLV_H

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "collective.h"
#include "communicator/mpix_comm.h"

#ifdef __cplusplus
extern "C"
{
#endif

// TODO : need to add hierarchical/locality-aware methods for alltoallv
enum AlltoallvMethod { ALLTOALLV_PAIRWISE, ALLTOALLV_NONBLOCKING, ALLTOALLV_BATCH, 
        ALLTOALLV_BATCH_ASYNC, ALLTOALLV_PMPI };
extern AlltoallvMethod mpix_alltoallv_implementation;

typedef int (*alltoallv_ftn)(const void*, const int*, const int*, MPI_Datatype,
void*, const int*, const int*, MPI_Datatype, MPIX_Comm*);

// Helper Functions
int alltoallv_pairwise(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int alltoallv_nonblocking(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int alltoallv_batch(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int alltoallv_batch_async(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm);


int alltoallv_pmpi(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm);

#ifdef __cplusplus
}
#endif

#endif
