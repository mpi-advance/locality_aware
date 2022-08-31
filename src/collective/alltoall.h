#ifndef MPI_ADVANCE_ALLTOALL_H
#define MPI_ADVANCE_ALLTOALL_H

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "utils.h"
#include "locality/topology.h"

#ifdef __cplusplus
extern "C"
{
#endif

// Helper Functions
int alltoall_bruck(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm);

#ifdef __cplusplus
}
#endif

#endif
