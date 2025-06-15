#ifndef MPI_ADVANCE_ALLREDUCE_H
#define MPI_ADVANCE_ALLREDUCE_H

#include <mpi.h>

#ifdef __cplusplus
extern "C"
{
#endif 

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

#ifdef __cplusplus
}
#endif

#endif // MPI_ADVANCE_ALLREDUCE_H