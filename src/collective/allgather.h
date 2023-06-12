#ifndef MPI_ADVANCE_ALLGATHER_H
#define MPI_ADVANCE_ALLGATHER_H

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "collective.h"
#include "locality/topology.h"

#ifdef __cplusplus
extern "C"
{
#endif

// Helper Functions
int allgather_bruck(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm);
int allgather_p2p(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm);
int allgather_ring(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm);
int allgather_loc_p2p(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int allgather_loc_bruck(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int allgather_loc_ring(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int allgather_hier_bruck(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int allgather_mult_hier_bruck(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);



#ifdef __cplusplus
}
#endif


#endif
