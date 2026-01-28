#ifndef MPI_ADVANCE_GPU_ALLREDUCE_INIT_H
#define MPI_ADVANCE_GPU_ALLREDUCE_INIT_H

#include "collective/allreduce_init.h"
#include "gpu_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(GPU_AWARE)
/** @brief A GPU-aware wrapper around provided ::allreduce_helper_ftn
 * @details This function assumes that the provided ::allreduce_helper_ftn is GPU aware, and
 * can handle GPU buffers. No extra behavior from the normal ::allreduce_helper_ftn.
 * @returns The result from the ::allreduce_helper_ftn
 **/
int gpu_aware_allreduce_init(allreduce_init_helper_ftn f,
                             const void* sendbuf,
                             void* recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op,
                             MPIL_Comm* comm,
                             MPIL_Info* info,
                             MPIL_Request** req_ptr);

int gpu_aware_allreduce_recursive_doubling_init(const void* sendbuf,
                                            void* recvbuf,
                                            int count,
                                            MPI_Datatype datatype,
                                            MPI_Op op,
                                            MPIL_Comm* comm,
                                            MPIL_Info* info,
                                            MPIL_Request** req_ptr);
int gpu_aware_allreduce_dissemination_loc_init(const void* sendbuf,
                                          void* recvbuf,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op,
                                          MPIL_Comm* comm,
                                          MPIL_Info* info,
                                          MPIL_Request** req_ptr);
int gpu_aware_allreduce_dissemination_ml_init(const void* sendbuf,
                                          void* recvbuf,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op,
                                          MPIL_Comm* comm,
                                          MPIL_Info* info,
                                          MPIL_Request** req_ptr);
#if defined(MPI4)
int gpu_aware_allreduce_pmpi_init(const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm,
                        MPIL_Info* info,
                        MPIL_Request** req_ptr);
#endif

#endif defined(GPU_AWARE)

int copy_to_cpu_allreduce_init(allreduce_init_helper_ftn f,
                          const void* sendbuf,
                          void* recvbuf,
                          int count,
                          MPI_Datatype datatype,
                          MPI_Op op,
                          MPIL_Comm* comm,
                          MPIL_Info* info,
                          MPIL_Request** req_ptr);
int copy_to_cpu_allreduce_recursive_doubling_init(const void* sendbuf,
                                           void* recvbuf,
                                           int count,
                                           MPI_Datatype datatype,
                                           MPI_Op op,
                                           MPIL_Comm* comm,
                                           MPIL_Info* info,
                                           MPIL_Request** req_ptr);
int copy_to_cpu_allreduce_dissemination_loc_init(const void* sendbuf,
                                          void* recvbuf,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op,
                                          MPIL_Comm* comm,
                                          MPIL_Info* info,
                                          MPIL_Request** req_ptr);
int copy_to_cpu_allreduce_dissemination_ml_init(const void* sendbuf,
                                         void* recvbuf,
                                         int count,
                                         MPI_Datatype datatype,
                                         MPI_Op op,
                                         MPIL_Comm* comm,
                                         MPIL_Info* info,
                                         MPIL_Request** req_ptr);
#if defined(MPI4)
int copy_to_cpu_allreduce_pmpi_init(const void* sendbuf,
                               void* recvbuf,
                               int count,
                               MPI_Datatype datatype,
                               MPI_Op op,
                               MPIL_Comm* comm,
                               MPIL_Info* info,
                               MPIL_Request** req_ptr);
#endif


#ifdef __cplusplus
}
#endif

#endif
