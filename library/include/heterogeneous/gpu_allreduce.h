#ifndef MPI_ADVANCE_GPU_ALLTOALL_H
#define MPI_ADVANCE_GPU_ALLTOALL_H

#include "collective/allreduce.h"
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
int gpu_aware_allreduce(allreduce_helper_ftn f,
                        const void* sendbuf,
                        void* recvbuf,
                        int count, 
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm); 

/** @brief Calls GPU-Aware PMPI**/
int gpu_aware_allreduce_pmpi(
                        const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm);
/** @brief Calls ::gpu_aware_allreduce with ::recursive doubling function**/
int gpu_aware_allreduce_recursive_doubling(
                        const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm);
/** @brief Calls ::gpu_aware_allreduce with ::dissemination node-aware function**/
int gpu_aware_allreduce_dissemination_loc(
                        const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm);
/** @brief Calls ::gpu_aware_allreduce with ::dissemination NUMA-aware function**/
int gpu_aware_allreduce_dissemination_ml(
                        const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm);
/** @brief Calls ::gpu_aware_allreduce with ::dissemination RADIX function**/
int gpu_aware_allreduce_dissemination_radix(
                        const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm);
#endif




/** @brief A GPU buffer variant wrapper around provided ::allreduce_helper_ftn
 * @details Unlike the GPU-aware variant, this version first allocates memory on the
 * hosts, copies the data from the GPU to the host, performs the requests allreduce, then
 * copies the final result back to the GPU. All parameters have the same requirements as
 * the ::allreduce_helper_ftn.
 * @returns The result from the ::allreduce_helper_ftn
 **/
int copy_to_cpu_allreduce(allreduce_helper_ftn f,
                          const void* sendbuf,
                          void* recvbuf,
                          int count,
                          MPI_Datatype datatype,
                          MPI_Op op,
                          MPIL_Comm* comm);

/** @brief Calls CopyToCPU PMPI**/
int copy_to_cpu_allreduce_pmpi(
                        const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm);
/** @brief Calls ::copy_to_cpu_allreduce with ::recursive doubling function**/
int copy_to_cpu_allreduce_recursive_doubling(
                        const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm);
/** @brief Calls ::copy_to_cpu_allreduce with ::dissemination node-aware function**/
int copy_to_cpu_allreduce_dissemination_loc(
                        const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm);
/** @brief Calls ::copy_to_cpu_allreduce with ::dissemination NUMA-aware function**/
int copy_to_cpu_allreduce_dissemination_ml(
                        const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm);
/** @brief Calls ::copy_to_cpu_allreduce with ::dissemination RADIX function**/
int copy_to_cpu_allreduce_dissemination_radix(
                        const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm);

#ifdef __cplusplus
}
#endif

#endif
