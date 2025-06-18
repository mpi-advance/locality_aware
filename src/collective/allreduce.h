#ifndef MPI_ADVANCE_ALLREDUCE_H
#define MPI_ADVANCE_ALLREDUCE_H

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "utils/utils.h"
#include "collective.h"
#include "communicator/mpix_comm.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef int (*allreduce_ftn)(const void*, void*, int, MPI_Datatype, MPI_Op, MPIX_Comm*);

int MPIX_Allreduce(const void* sendbuf,
        void* recvbuf, 
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm* comm);

// Reduce_scatter + alltoall + allgather
int allreduce_lane(const void* sendbuf,
        void* recvbuf, 
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm* comm);

// Alltoall + alltoall
int allreduce_loc(const void* sendbuf,
        void* recvbuf, 
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm* comm);

#ifdef __cplusplus
}
#endif

#endif
