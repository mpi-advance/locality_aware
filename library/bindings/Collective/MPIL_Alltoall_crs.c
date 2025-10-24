#include "locality_aware.h"
#include "neighborhood/alltoall_crs.h"

int MPIL_Alltoall_crs(const int send_nnz,
                      const int* dest,
                      const int sendcount,
                      MPI_Datatype sendtype,
                      const void* sendvals,
                      int* recv_nnz,
                      int** src_ptr,
                      int recvcount,
                      MPI_Datatype recvtype,
                      void** recvvals_ptr,
                      MPIL_Info* xinfo,
                      MPIL_Comm* xcomm)
{
    alltoall_crs_ftn method;
    switch (mpil_alltoall_crs_implementation)
    {
        case ALLTOALL_CRS_PERSONALIZED:
            method = alltoall_crs_personalized;
            break;
        case ALLTOALL_CRS_PERSONALIZED_LOC:
            method = alltoall_crs_personalized_loc;
            break;
        case ALLTOALL_CRS_RMA:
            method = alltoall_crs_rma;
            break;
        case ALLTOALL_CRS_NONBLOCKING:
            method = alltoall_crs_nonblocking;
            break;
        case ALLTOALL_CRS_NONBLOCKING_LOC:
            method = alltoall_crs_nonblocking_loc;
            break;
        default:
            method = alltoall_crs_personalized;
            break;
    }
    return method(send_nnz,
                  dest,
                  sendcount,
                  sendtype,
                  sendvals,
                  recv_nnz,
                  src_ptr,
                  recvcount,
                  recvtype,
                  recvvals_ptr,
                  xinfo,
                  xcomm);
}