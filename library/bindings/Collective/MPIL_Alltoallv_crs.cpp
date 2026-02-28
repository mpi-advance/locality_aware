#include "locality_aware.h"
#include "neighborhood/alltoall_crs.h"

int MPIL_Alltoallv_crs(const int send_nnz,
                       const int send_size,
                       const int* dest,
                       const int* sendcounts,
                       const int* sdispls,
                       MPI_Datatype sendtype,
                       const void* sendvals,
                       int* recv_nnz,
                       int* recv_size,
                       int** src_ptr,
                       int** recvcounts_ptr,
                       int** rdispls_ptr,
                       MPI_Datatype recvtype,
                       void** recvvals_ptr,
                       MPIL_Info* xinfo,
                       MPIL_Comm* xcomm)
{
    alltoallv_crs_ftn method;

    switch (mpil_alltoallv_crs_implementation)
    {
        case ALLTOALLV_CRS_NONBLOCKING:
            method = alltoallv_crs_nonblocking;
            break;
        case ALLTOALLV_CRS_NONBLOCKING_LOC:
            method = alltoallv_crs_nonblocking_loc;
            break;
        case ALLTOALLV_CRS_PERSONALIZED:
            method = alltoallv_crs_personalized;
            break;
        case ALLTOALLV_CRS_PERSONALIZED_LOC:
            method = alltoallv_crs_personalized_loc;
            break;
        default:
            method = alltoallv_crs_personalized;
            break;
    }

    return method(send_nnz,
                  send_size,
                  dest,
                  sendcounts,
                  sdispls,
                  sendtype,
                  sendvals,
                  recv_nnz,
                  recv_size,
                  src_ptr,
                  recvcounts_ptr,
                  rdispls_ptr,
                  recvtype,
                  recvvals_ptr,
                  xinfo,
                  xcomm);
}
