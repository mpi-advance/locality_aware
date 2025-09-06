#ifndef MPI_ADVANCE_ALLREDUCE_H
#define MPI_ADVANCE_ALLREDUCE_H

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils/utils.h"
#include "collective.h"
#include "communicator/mpix_comm.h"

#ifdef __cplusplus
extern "C"
{
#endif 

int allreduce_multileader(const void *sendbuf, 
        void *recvbuf,
        const int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm comm);

int allreduce_hierarchical(const void *sendbuf,
        void *recvbuf,
        const int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm comm);

int allreduce_node_aware(const void *sendbuf,
        void *recvbuf,
        const int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm);

int allreduce_locality_aware(const void *sendbuf,
        void *recvbuf,
        const int count,
        MPI_Datatype datatype,
        MPI_Op op,
		MPIX_Comm comm);

int allreduce_multileader_locality(
    const void* sendbuf,
    void* recvbuf,
    const int count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPIX_Comm comm);

#ifdef __cplusplus
}
#endif

#endif // MPI_ADVANCE_ALLREDUCE_H
