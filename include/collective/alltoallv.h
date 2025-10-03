#ifndef MPI_ADVANCE_ALLTOALLV_H
#define MPI_ADVANCE_ALLTOALLV_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

//#include "collective.h"
#include "../communicator/mpil_comm.h"

#ifdef __cplusplus
extern "C" {
#endif

// TODO : need to add hierarchical/locality-aware methods for alltoallv
enum AlltoallvMethod
{
    ALLTOALLV_PAIRWISE,
    ALLTOALLV_NONBLOCKING,
    ALLTOALLV_BATCH,
    ALLTOALLV_BATCH_ASYNC,
    ALLTOALLV_PMPI
};
extern enum AlltoallvMethod mpil_alltoallv_implementation;

typedef int (*alltoallv_ftn)(const void*,
                             const int*,
                             const int*,
                             MPI_Datatype,
                             void*,
                             const int*,
                             const int*,
                             MPI_Datatype,
                             MPIL_Comm*);

// Helper Functions
int alltoallv_pairwise(const void* sendbuf,
                       const int sendcounts[],
                       const int sdispls[],
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcounts[],
                       const int rdispls[],
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm);
int alltoallv_nonblocking(const void* sendbuf,
                          const int sendcounts[],
                          const int sdispls[],
                          MPI_Datatype sendtype,
                          void* recvbuf,
                          const int recvcounts[],
                          const int rdispls[],
                          MPI_Datatype recvtype,
                          MPIL_Comm* comm);
int alltoallv_batch(const void* sendbuf,
                    const int sendcounts[],
                    const int sdispls[],
                    MPI_Datatype sendtype,
                    void* recvbuf,
                    const int recvcounts[],
                    const int rdispls[],
                    MPI_Datatype recvtype,
                    MPIL_Comm* comm);
int alltoallv_batch_async(const void* sendbuf,
                          const int sendcounts[],
                          const int sdispls[],
                          MPI_Datatype sendtype,
                          void* recvbuf,
                          const int recvcounts[],
                          const int rdispls[],
                          MPI_Datatype recvtype,
                          MPIL_Comm* comm);

int alltoallv_pmpi(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm);

#ifdef __cplusplus
}
#endif

#endif
