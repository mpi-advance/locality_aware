#ifndef MPI_ADVANCE_COLLECTIVES_H
#define MPI_ADVANCE_COLLECTIVES_H

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "utils.h"

#ifdef __cplusplus
extern "C"
{
#endif

int PMPI_Alltoallv(const void* sendbuf,
        const int* sendcounts,
        const int* sdispls,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int* recvcounts,
        const int* rdispls,
        MPI_Datatype recvtype,
        MPI_Comm comm);

#ifdef __cplusplus
}
#endif


#endif
