#ifndef MPI_ADVANCE_ALLGATHER_H
#define MPI_ADVANCE_ALLGATHER_H

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
int MPIX_Allgather_bruck(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm);
int MPIX_Allgather_p2p(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm);
int MPIX_Allgather_ring(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm);

// Locality helper functions
int MPIX_Allgather_bruck_locality(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int MPIX_Allgather_p2p_locality(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int MPIX_Allgather_ring_locality(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int MPIX_Allgather_hier_bruck(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm);
int MPIX_Allgather_mult_hier_bruck(const void* sendbuf,
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
