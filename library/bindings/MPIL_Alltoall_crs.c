#include "../../include/neighborhood/sparse_coll.h"

//#include <stdlib.h>
//#include <string.h>

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
    return alltoall_crs_personalized(send_nnz,
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