#ifndef MPI_ADVANCE_COLLECTIVES_H
#define MPI_ADVANCE_COLLECTIVES_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "../communicator/mpil_comm.h"


//#include "../utils/utils.h"

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Alltoall(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* comm);

int MPIL_Alltoallv(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm);

int MPI_Alltoallv(const void* sendbuf,
                  const int sendcounts[],
                  const int sdispls[],
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcounts[],
                  const int rdispls[],
                  MPI_Datatype recvtype,
                  MPI_Comm comm); 

enum AlltoallMethod
{
    ALLTOALL_PAIRWISE,
    ALLTOALL_NONBLOCKING,
    ALLTOALL_HIERARCHICAL_PAIRWISE,
    ALLTOALL_HIERARCHICAL_NONBLOCKING,
    ALLTOALL_MULTILEADER_PAIRWISE,
    ALLTOALL_MULTILEADER_NONBLOCKING,
    ALLTOALL_NODE_AWARE_PAIRWISE,
    ALLTOALL_NODE_AWARE_NONBLOCKING,
    ALLTOALL_LOCALITY_AWARE_PAIRWISE,
    ALLTOALL_LOCALITY_AWARE_NONBLOCKING,
    ALLTOALL_MULTILEADER_LOCALITY_PAIRWISE,
    ALLTOALL_MULTILEADER_LOCALITY_NONBLOCKING,
    ALLTOALL_PMPI
};
extern enum AlltoallMethod mpil_alltoall_implementation;

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

#ifdef __cplusplus
}
#endif

#endif
