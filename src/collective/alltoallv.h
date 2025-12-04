#ifndef MPI_ADVANCE_ALLTOALLV_H
#define MPI_ADVANCE_ALLTOALLV_H

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "/g/g92/enamug/clean/GPU_locality_aware/locality_aware/src/utils.h"
#include "/g/g92/enamug/clean/GPU_locality_aware/locality_aware/src/collective/collective.h"
#include "/g/g92/enamug/clean/GPU_locality_aware/locality_aware/src/locality/topology.h"
#include "/g/g92/enamug/clean/GPU_locality_aware/locality_aware/src/persistent/persistent.h"
#ifdef __cplusplus
extern "C"
{
#endif

// Helper Functions
int alltoallv_pairwise(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm);
int alltoallv_pairwise_log2(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm);
int alltoallv_nonblocking(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm);

int alltoallv_init(const void* sendbuf,
       const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
       const int recvcounts[],
       const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr);

int alltoallv_nonblocking_init(const void* sendbuf,
       const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
       const int recvcounts[],
       const int rdispls[],
        MPI_Datatype recvtype,
        //MPI_Comm comm,
        MPIX_Comm* xcomm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr);

int alltoallv_rma_winflush(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm);

int alltoallv_rma_winfence(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm);

int alltoallv_rma_winfence_init(const void* sendbuf,
                const int sendcounts[],
                const int sdispls[],
                MPI_Datatype sendtype,
                void* recvbuf,
                const int recvcounts[],
                const int rdispls[],
                MPI_Datatype recvtype,
                MPIX_Comm* xcomm,
                MPIX_Info* xinfo,
                MPIX_Request** request_ptr);

int alltoallv_rma_winfence_init_han(const void* sendbuf,
                const int sendcounts[],
                const int sdispls[],
                MPI_Datatype sendtype,
                void* recvbuf,
                const int recvcounts[],
                const int rdispls[],
                MPI_Datatype recvtype,
                MPIX_Comm* xcomm,
                MPIX_Info* xinfo,
                MPIX_Request** request_ptr);                

int alltoallv_rma_lock_init(const void* sendbuf,
                        const int sendcounts[],
                        const int sdispls[],
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        const int recvcounts[],
                        const int rdispls[],
                        MPI_Datatype recvtype,
                        MPIX_Comm* xcomm,
                        MPIX_Info* xinfo,
                        MPIX_Request** request_ptr);
                  

int alltoallv_rma_lock_init_han(const void* sendbuf,
                        const int sendcounts[],
                        const int sdispls[],
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        const int recvcounts[],
                        const int rdispls[],
                        MPI_Datatype recvtype,
                        MPIX_Comm* xcomm,
                        MPIX_Info* xinfo,
                        MPIX_Request** request_ptr);


int alltoallv_rma_winlock(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm);  

int alltoallv_rma_newly_winlock(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm);  


       /*
int alltoallv_rma_init(const void* sendbuf,
        const int* sendcounts,
        const int* sdispls,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int* recvcounts,
        const int* rdispls,
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr);

        */


int alltoallv_rma_init(const void* sendbuf,
        const int* sendcounts,
        const int* sdispls,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int* recvcounts,
        const int* rdispls,
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr);




int alltoallv_pairwise_init(const void* sendbuf,
       const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
       const int recvcounts[],
       const int rdispls[],
        MPI_Datatype recvtype,
        //MPI_Comm comm,
        MPIX_Comm* xcomm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr);




int alltoallv_pairwise_nonblocking(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm);
int alltoallv_waitany(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm);
int alltoallv_pairwise_nonblocking_log2(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm);






int alltoallv_pairwise_loc(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm);


#ifdef __cplusplus
}
#endif

#endif
