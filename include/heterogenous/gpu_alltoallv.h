#ifndef MPI_ADVANCE_GPU_ALLTOALLV_H
#define MPI_ADVANCE_GPU_ALLTOALLV_H

#include "collective/alltoallv.h"
#include "collective/collective.h"

#ifdef __cplusplus
extern "C" {
#endif

int gpu_aware_alltoallv(alltoallv_ftn f,
                        const void* sendbuf,
                        const int sendcounts[],
                        const int sdispls[],
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        const int recvcounts[],
                        const int rdispls[],
                        MPI_Datatype recvtype,
                        MPIL_Comm* comm);

int copy_to_cpu_alltoallv(alltoallv_ftn f,
                          const void* sendbuf,
                          const int sendcounts[],
                          const int sdispls[],
                          MPI_Datatype sendtype,
                          void* recvbuf,
                          const int recvcounts[],
                          const int rdispls[],
                          MPI_Datatype recvtype,
                          MPIL_Comm* comm);

int gpu_aware_alltoallv_pairwise(const void* sendbuf,
                                 const int sendcounts[],
                                 const int sdispls[],
                                 MPI_Datatype sendtype,
                                 void* recvbuf,
                                 const int recvcounts[],
                                 const int rdispls[],
                                 MPI_Datatype recvtype,
                                 MPIL_Comm* comm);

int gpu_aware_alltoallv_nonblocking(const void* sendbuf,
                                    const int sendcounts[],
                                    const int sdispls[],
                                    MPI_Datatype sendtype,
                                    void* recvbuf,
                                    const int recvcounts[],
                                    const int rdispls[],
                                    MPI_Datatype recvtype,
                                    MPIL_Comm* comm);

int gpu_aware_alltoallv_batch(const void* sendbuf,
                              const int sendcounts[],
                              const int sdispls[],
                              MPI_Datatype sendtype,
                              void* recvbuf,
                              const int recvcounts[],
                              const int rdispls[],
                              MPI_Datatype recvtype,
                              MPIL_Comm* comm);

int gpu_aware_alltoallv_batch_async(const void* sendbuf,
                                    const int sendcounts[],
                                    const int sdispls[],
                                    MPI_Datatype sendtype,
                                    void* recvbuf,
                                    const int recvcounts[],
                                    const int rdispls[],
                                    MPI_Datatype recvtype,
                                    MPIL_Comm* comm);

int copy_to_cpu_alltoallv_pairwise(const void* sendbuf,
                                   const int sendcounts[],
                                   const int sdispls[],
                                   MPI_Datatype sendtype,
                                   void* recvbuf,
                                   const int recvcounts[],
                                   const int rdispls[],
                                   MPI_Datatype recvtype,
                                   MPIL_Comm* comm);

int copy_to_cpu_alltoallv_nonblocking(const void* sendbuf,
                                      const int sendcounts[],
                                      const int sdispls[],
                                      MPI_Datatype sendtype,
                                      void* recvbuf,
                                      const int recvcounts[],
                                      const int rdispls[],
                                      MPI_Datatype recvtype,
                                      MPIL_Comm* comm);

int copy_to_cpu_alltoallv_batch(const void* sendbuf,
                                const int sendcounts[],
                                const int sdispls[],
                                MPI_Datatype sendtype,
                                void* recvbuf,
                                const int recvcounts[],
                                const int rdispls[],
                                MPI_Datatype recvtype,
                                MPIL_Comm* comm);

int copy_to_cpu_alltoallv_batch_async(const void* sendbuf,
                                      const int sendcounts[],
                                      const int sdispls[],
                                      MPI_Datatype sendtype,
                                      void* recvbuf,
                                      const int recvcounts[],
                                      const int rdispls[],
                                      MPI_Datatype recvtype,
                                      MPIL_Comm* comm);

#ifdef OPENMP
#include <omp.h>
int threaded_alltoallv_pairwise(const void* sendbuf,
                                const int sendcounts[],
                                const int sdispls[],
                                MPI_Datatype sendtype,
                                void* recvbuf,
                                const int recvcounts[],
                                const int rdispls[],
                                MPI_Datatype recvtype,
                                MPIL_Comm* comm);

int threaded_alltoallv_nonblocking(const void* sendbuf,
                                   const int sendcounts[],
                                   const int sdispls[],
                                   MPI_Datatype sendtype,
                                   void* recvbuf,
                                   const int recvcounts[],
                                   const int rdispls[],
                                   MPI_Datatype recvtype,
                                   MPIL_Comm* comm);
#endif

#ifdef __cplusplus
}
#endif

#endif
