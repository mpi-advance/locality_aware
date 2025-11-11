#ifndef MPI_ADVANCE_GPU_ALLTOALL_H
#define MPI_ADVANCE_GPU_ALLTOALL_H

#include "collective/alltoall.h"
#include "gpu_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief gpu wrapper around alltoall_ftn f 
	@details
		allocates host memory for send and recv buffers.  
	    has same arguments as alltoall_helper function.
	@returns ierr
**/
int gpu_aware_alltoall(alltoall_ftn f,
                       const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm);
			
/** @brief offloads communication off gpu into unpinned cpu memory, operates and writes back.  
	@details
		allocates host memory for send and recv buffers.
		copy from gpu to cpu
	    has same arguments as alltoall_helper function.
	@returns ierr
**/		
int copy_to_cpu_alltoall(alltoall_ftn f,
                         const void* sendbuf,
                         const int sendcount,
                         MPI_Datatype sendtype,
                         void* recvbuf,
                         const int recvcount,
                         MPI_Datatype recvtype,
                         MPIL_Comm* comm);

/** @brief calls gpu_aware_alltoall with pairwise_helper function**/
int gpu_aware_alltoall_pairwise(const void* sendbuf,
                                const int sendcount,
                                MPI_Datatype sendtype,
                                void* recvbuf,
                                const int recvcount,
                                MPI_Datatype recvtype,
                                MPIL_Comm* comm);
/** @brief calls gpu_aware_alltoall with nonblocking_helper function**/
int gpu_aware_alltoall_nonblocking(const void* sendbuf,
                                   const int sendcount,
                                   MPI_Datatype sendtype,
                                   void* recvbuf,
                                   const int recvcount,
                                   MPI_Datatype recvtype,
                                   MPIL_Comm* comm);
/** @brief calls copy_to_cpu_alltoall with pairwise_helper function**/							   
int copy_to_cpu_alltoall_pairwise(const void* sendbuf,
                                  const int sendcount,
                                  MPI_Datatype sendtype,
                                  void* recvbuf,
                                  const int recvcount,
                                  MPI_Datatype recvtype,
                                  MPIL_Comm* comm);
/** @brief calls copy_to_cpu_alltoall with nonblocking_helper function**/	
int copy_to_cpu_alltoall_nonblocking(const void* sendbuf,
                                     const int sendcount,
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcount,
                                     MPI_Datatype recvtype,
                                     MPIL_Comm* comm);

//#ifdef OPENMP
//#include <omp.h>
/** @brief Untested function using OpenMP to divide GPU buffers and proc sends. 
	@details
	    \todo Not currently tested or included in switch
		seperates buffers among openMP threads, each thread calls Sendrecv?
**/
int threaded_alltoall_pairwise(const void* sendbuf,
                               const int sendcount,
                               MPI_Datatype sendtype,
                               void* recvbuf,
                               const int recvcount,
                               MPI_Datatype recvtype,
                               MPIL_Comm* comm);

/** @brief Untested function using OpenMP to divide GPU buffers and proc sends. 
	@details
	    \todo Not currently tested or included in switch
		seperates buffers among openMP threads, each thread calls Isend and Irecv?
**/
int threaded_alltoall_nonblocking(const void* sendbuf,
                                  const int sendcount,
                                  MPI_Datatype sendtype,
                                  void* recvbuf,
                                  const int recvcount,
                                  MPI_Datatype recvtype,
                                  MPIL_Comm* comm);
//#endif

#ifdef __cplusplus
}
#endif

#endif
