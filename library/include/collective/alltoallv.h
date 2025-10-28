#ifndef MPI_ADVANCE_ALLTOALLV_H
#define MPI_ADVANCE_ALLTOALLV_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "communicator/MPIL_Comm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*alltoallv_ftn)(const void*,
                             const int*,
                             const int*,
                             MPI_Datatype,
                             void*,
                             const int*,
                             const int*,
                             MPI_Datatype,
                             MPIL_Comm*);

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
