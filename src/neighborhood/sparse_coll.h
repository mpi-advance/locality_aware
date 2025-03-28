#ifndef MPI_ADVANCE_SPARSE_COLL_H
#define MPI_ADVANCE_SPARSE_COLL_H

#include "mpi.h"
#include "locality/locality_comm.h"
#include "locality/topology.h"
#include "utils/utils.h"

// Declarations of C++ methods
#ifdef __cplusplus
extern "C"
{
#endif
 

int MPIX_Alltoall_crs(
        int send_nnz,
        int* dest,
        int sendcount,
        MPI_Datatype sendtype,
        void* sendvals,
        int* recv_nnz,
        int** src_ptr,
        int recvcount,
        MPI_Datatype recvtype,
        void** recvvals_ptr,
        MPIX_Info* xinfo,
        MPIX_Comm* xcomm
        );

int MPIX_Alltoallv_crs(
        int send_nnz,
        int send_size,
        int* dest,
        int* sendcounts,
        int* sdispls,
        MPI_Datatype sendtype,
        void* sendvals,
        int* recv_nnz,
        int* recv_size,
        int** src_ptr,
        int** recvcounts_ptr,
        int** rdispls_ptr,
        MPI_Datatype recvtype,
        void** recvvals_ptr,
        MPIX_Info* xinfo,
        MPIX_Comm* comm);


int alltoall_crs_rma(int send_nnz, int* dest, int sendcount,
        MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int** src, int recvcount, MPI_Datatype recvtype,
        void** recvvals, MPIX_Info* xinfo, MPIX_Comm* comm);

int alltoall_crs_personalized(int send_nnz, int* dest, int sendcount,
        MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int** src, int recvcount, MPI_Datatype recvtype,
        void** recvvals, MPIX_Info* xinfo, MPIX_Comm* comm);

int alltoall_crs_personalized_loc(int send_nnz, int* dest, int sendcount,
        MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int** src, int recvcount, MPI_Datatype recvtype,
        void** recvvals, MPIX_Info* xinfo, MPIX_Comm* comm);

int alltoall_crs_nonblocking(int send_nnz, int* dest, int sendcount,
        MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int** src, int recvcount, MPI_Datatype recvtype,
        void** recvvals, MPIX_Info* xinfo, MPIX_Comm* comm);

int alltoall_crs_nonblocking_loc(int send_nnz, int* dest, int sendcount,
        MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int** src, int recvcount, MPI_Datatype recvtype,
        void** recvvals, MPIX_Info* xinfo, MPIX_Comm* comm);



int alltoallv_crs_personalized(int send_nnz, int send_size, int* dest, int* sendcounts,
        int* sdispls, MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int* recv_size, int** src_ptr, int** recvcounts_ptr, 
        int** rdispls_ptr, MPI_Datatype recvtype, void** recvvals_ptr, MPIX_Info* xinfo, 
        MPIX_Comm* comm);

int alltoallv_crs_personalized_loc(int send_nnz, int send_size, int* dest, int* sendcounts,
        int* sdispls, MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int* recv_size, int** src_ptr, int** recvcounts_ptr, 
        int** rdispls_ptr, MPI_Datatype recvtype, void** recvvals_ptr, MPIX_Info* xinfo, 
        MPIX_Comm* comm);

int alltoallv_crs_nonblocking(int send_nnz, int send_size, int* dest, int* sendcounts,
        int* sdispls, MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int* recv_size, int** src_ptr, int** recvcounts_ptr, 
        int** rdispls_ptr, MPI_Datatype recvtype, void** recvvals_ptr, MPIX_Info* xinfo, 
        MPIX_Comm* comm);

int alltoallv_crs_nonblocking_loc(int send_nnz, int send_size, int* dest, int* sendcounts,
        int* sdispls, MPI_Datatype sendtype, void* sendvals,
        int* recv_nnz, int* recv_size, int** src_ptr, int** recvcounts_ptr, 
        int** rdispls_ptr, MPI_Datatype recvtype, void** recvvals_ptr, MPIX_Info* xinfo, 
        MPIX_Comm* comm);



#ifdef __cplusplus
}



#endif


#endif
