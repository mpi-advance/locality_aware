#ifndef MPI_ADVANCE_SPARSE_COLL_H
#define MPI_ADVANCE_SPARSE_COLL_H

#include "../communicator/locality_comm.h"
#include "../communicator/mpil_comm.h"
#include "mpi.h"
#include "../utils/utils.h"

// Declarations of C++ methods
#ifdef __cplusplus
extern "C" {
#endif

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
                      MPIL_Comm* xcomm);

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
                       MPIL_Comm* comm);

int alltoall_crs_rma(const int send_nnz,
                     const int* dest,
                     const int sendcount,
                     MPI_Datatype sendtype,
                     const void* sendvals,
                     int* recv_nnz,
                     int** src,
                     int recvcount,
                     MPI_Datatype recvtype,
                     void** recvvals,
                     MPIL_Info* xinfo,
                     MPIL_Comm* comm);

int alltoall_crs_personalized(const int send_nnz,
                              const int* dest,
                              const int sendcount,
                              MPI_Datatype sendtype,
                              const void* sendvals,
                              int* recv_nnz,
                              int** src,
                              int recvcount,
                              MPI_Datatype recvtype,
                              void** recvvals,
                              MPIL_Info* xinfo,
                              MPIL_Comm* comm);

int alltoall_crs_personalized_loc(const int send_nnz,
                                  const int* dest,
                                  const int sendcount,
                                  MPI_Datatype sendtype,
                                  const void* sendvals,
                                  int* recv_nnz,
                                  int** src,
                                  int recvcount,
                                  MPI_Datatype recvtype,
                                  void** recvvals,
                                  MPIL_Info* xinfo,
                                  MPIL_Comm* comm);

int alltoall_crs_nonblocking(const int send_nnz,
                             const int* dest,
                             const int sendcount,
                             MPI_Datatype sendtype,
                             const void* sendvals,
                             int* recv_nnz,
                             int** src,
                             int recvcount,
                             MPI_Datatype recvtype,
                             void** recvvals,
                             MPIL_Info* xinfo,
                             MPIL_Comm* comm);

int alltoall_crs_nonblocking_loc(const int send_nnz,
                                 const int* dest,
                                 const int sendcount,
                                 MPI_Datatype sendtype,
                                 const void* sendvals,
                                 int* recv_nnz,
                                 int** src,
                                 int recvcount,
                                 MPI_Datatype recvtype,
                                 void** recvvals,
                                 MPIL_Info* xinfo,
                                 MPIL_Comm* comm);

int alltoallv_crs_personalized(const int send_nnz,
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
                               MPIL_Comm* comm);

int alltoallv_crs_personalized_loc(const int send_nnz,
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
                                   MPIL_Comm* comm);

int alltoallv_crs_nonblocking(const int send_nnz,
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
                              MPIL_Comm* comm);

int alltoallv_crs_nonblocking_loc(const int send_nnz,
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
                                  MPIL_Comm* comm);

#ifdef __cplusplus
}

#endif

#endif
