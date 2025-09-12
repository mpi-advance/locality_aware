#ifndef MPI_ADVANCE_ALLREDUCE_H
#define MPI_ADVANCE_ALLREDUCE_H

#include <mpi.h>

#ifdef __cplusplus
extern "C"
{
#endif

int allgather_multileader(const void* sendbuf, 
                          int sendcount, 
                          MPI_Datatype sendtype,
                          void *recvbuf, 
                          int recvcount,
                          MPI_Datatype recvtype, 
                          MPIX_Comm comm);


int allgather_hierarchical(const void* sendbuf,
                           int sendcount,
                           MPI_Datatype sendtype,
                           void *recvbuf,
                           int recvcount,
                           MPI_Datatype recvtype,
                           MPIX_Comm comm);

int allgather_locality_aware(const void* sendbuf,
                             int sendcount,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             int recvcount,
                             MPI_Datatype recvtype,
                             MPIX_Comm comm);

int allgather_node_aware(const void* sendbuf,
                         int sendcount,
                         MPI_Datatype sendtype,
                         void* recvbuf,
                         int recvcount,
                         MPI_Datatype recvtype,
                         MPIX_Comm comm);

int allgather_multileader_locality_aware(const void* sendbuf,
                                         int sendcount,
                                         MPI_Datatype sendtype,
                                         void* recvbuf,
                                         int recvcount,
                                         MPI_Datatype recvtype,
                                         MPIX_Comm comm);


#ifdef 
__cplusplus
}
#endif

#endif // MPI_ADVANCE_ALLREDUCE_H