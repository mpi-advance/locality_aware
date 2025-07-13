#ifndef MPI_ADVANCE_ALLTOALL_H
#define MPI_ADVANCE_ALLTOALL_H

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

// TODO : need to add batch/batch asynch as underlying options for Alltoall
enum AlltoallMethod { ALLTOALL_PAIRWISE, ALLTOALL_NONBLOCKING, ALLTOALL_HIERARCHICAL_PAIRWISE, 
        ALLTOALL_HIERARCHICAL_NONBLOCKING, ALLTOALL_MULTILEADER_PAIRWISE, ALLTOALL_MULTILEADER_NONBLOCKING, 
        ALLTOALL_NODE_AWARE_PAIRWISE, ALLTOALL_NODE_AWARE_NONBLOCKING, ALLTOALL_LOCALITY_AWARE_PAIRWISE,
        ALLTOALL_LOCALITY_AWARE_NONBLOCKING, ALLTOALL_MULTILEADER_LOCALITY_PAIRWISE, 
        ALLTOALL_MULTILEADER_LOCALITY_NONBLOCKING, ALLTOALL_PMPI };
extern AlltoallMethod mpix_alltoall_implementation;


typedef int (*alltoall_ftn)(const void*, const int, MPI_Datatype, void*, const int, MPI_Datatype, MPIX_Comm*);
typedef int (*alltoall_helper_ftn)(const void*, const int, MPI_Datatype, void*, const int, MPI_Datatype, MPI_Comm, int tag);

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


int alltoall_hierarchical_pairwise(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int alltoall_hierarchical_nonblocking(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int alltoall_multileader_pairwise(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int alltoall_multileader_nonblocking(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);



int alltoall_node_aware_pairwise(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int alltoall_node_aware_nonblocking(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int alltoall_locality_aware_pairwise(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int alltoall_locality_aware_nonblocking(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);



int alltoall_multileader_locality_pairwise(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int alltoall_multileader_locality_nonblocking(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);



// Calls underlying MPI implementation
int alltoall_pmpi(const void* sendbuf,
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
