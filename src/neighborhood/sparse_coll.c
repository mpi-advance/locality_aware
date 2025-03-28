#include "sparse_coll.h"
#include <stdlib.h>
#include <string.h>

int MPIX_Alltoall_crs(
        int send_nnz,
        int* dest,
        int sendcount,
        MPI_Datatype sendtype,
        void* sendvals,
        int* recv_nnz,
        int** src,
        int recvcount,
        MPI_Datatype recvtype,
        void** recvvals,
        MPIX_Info* xinfo,
        MPIX_Comm* xcomm)
{
    return alltoall_crs_personalized(send_nnz, dest, sendcount, sendtype, sendvals,
            recv_nnz, src, recvcount, recvtype, recvvals, xinfo, xcomm);
}

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
        int** src,
        int** recvcounts,
        int** rdispls,
        MPI_Datatype recvtype,
        void** recvvals,
        MPIX_Info* xinfo,
        MPIX_Comm* xcomm)
{
    return alltoallv_crs_personalized(send_nnz, send_size, dest, sendcounts, sdispls,
            sendtype, sendvals, recv_nnz, recv_size, src, recvcounts, 
            rdispls, recvtype, recvvals, xinfo, xcomm);
}

