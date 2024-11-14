#ifndef MPI_ADVANCE_ALLTOALL_H
#define MPI_ADVANCE_ALLTOALL_H

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "utils/utils.h"
#include "collective.h"
#include "locality/topology.hpp"

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
        MPIX_Comm* comm);
int alltoall_nonblocking(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);


#ifdef __cplusplus
}
#endif

#endif
