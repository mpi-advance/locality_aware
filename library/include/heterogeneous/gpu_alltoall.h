#ifndef MPI_ADVANCE_GPU_ALLTOALL_H
#define MPI_ADVANCE_GPU_ALLTOALL_H

//#include "../collective/alltoall.h"
//#include "../../../include/communicator/mpil_comm.h"

#include "../collective/alltoall.h"
#include "../../../include/communicator/mpil_comm.h"


#ifdef __cplusplus
extern "C" {
#endif

int gpu_aware_alltoall(alltoall_ftn f,
                       const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm);
int copy_to_cpu_alltoall(alltoall_ftn f,
                         const void* sendbuf,
                         const int sendcount,
                         MPI_Datatype sendtype,
                         void* recvbuf,
                         const int recvcount,
                         MPI_Datatype recvtype,
                         MPIL_Comm* comm);

int gpu_aware_alltoall_pairwise(const void* sendbuf,
                                const int sendcount,
                                MPI_Datatype sendtype,
                                void* recvbuf,
                                const int recvcount,
                                MPI_Datatype recvtype,
                                MPIL_Comm* comm);
int gpu_aware_alltoall_nonblocking(const void* sendbuf,
                                   const int sendcount,
                                   MPI_Datatype sendtype,
                                   void* recvbuf,
                                   const int recvcount,
                                   MPI_Datatype recvtype,
                                   MPIL_Comm* comm);
int copy_to_cpu_alltoall_pairwise(const void* sendbuf,
                                  const int sendcount,
                                  MPI_Datatype sendtype,
                                  void* recvbuf,
                                  const int recvcount,
                                  MPI_Datatype recvtype,
                                  MPIL_Comm* comm);
int copy_to_cpu_alltoall_nonblocking(const void* sendbuf,
                                     const int sendcount,
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcount,
                                     MPI_Datatype recvtype,
                                     MPIL_Comm* comm);

#ifdef OPENMP
#include <omp.h>
int threaded_alltoall_pairwise(const void* sendbuf,
                               const int sendcount,
                               MPI_Datatype sendtype,
                               void* recvbuf,
                               const int recvcount,
                               MPI_Datatype recvtype,
                               MPIL_Comm* comm);

int threaded_alltoall_nonblocking(const void* sendbuf,
                                  const int sendcount,
                                  MPI_Datatype sendtype,
                                  void* recvbuf,
                                  const int recvcount,
                                  MPI_Datatype recvtype,
                                  MPIL_Comm* comm);
#endif

#ifdef __cplusplus
}
#endif

#endif
