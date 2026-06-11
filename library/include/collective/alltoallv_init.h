#ifndef MPI_ADVANCE_ALLTOALLV_INIT_H
#define MPI_ADVANCE_ALLTOALLV_INIT_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "communicator/MPIL_Comm.hpp"
#include "communicator/MPIL_Info.h"
#include "persistent/MPIL_Request.h"

#ifdef __cplusplus
extern "C" {
#endif
/** @brief Function pointer to alltoallv implementation
 * @details 
 * Uses the parameters of standard MPI_Alltoallv API, except replacing MPI_Comm with MPIL_Comm.
 * Most of the behavior is derived from internal parameters in MPIL_Comm.
 *  
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** persistent request object
 **/
typedef int (*alltoallv_init_ftn)(const void*,
                             const int*,
                             const int*,
                             MPI_Datatype,
                             void*,
                             const int*,
                             const int*,
                             MPI_Datatype,
                             MPIL_Comm*,
                             MPIL_Info*,
                             MPIL_Request**);
							 
/** @brief Uses Sendrecv to do the alltoallv
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** persistent request object
 **/
int alltoallv_init_pairwise(const void* sendbuf,
                       const int sendcounts[],
                       const int sdispls[],
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcounts[],
                       const int rdispls[],
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr);

/** @brief Uses Isend and Irecv to do the alltoallv
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] info MPIL_Info used for hints
 * @param [out] req_ptr MPIL_Request** persistent request object
 **/
int alltoallv_init_nonblocking(const void* sendbuf,
                          const int sendcounts[],
                          const int sdispls[],
                          MPI_Datatype sendtype,
                          void* recvbuf,
                          const int recvcounts[],
                          const int rdispls[],
                          MPI_Datatype recvtype,
                          MPIL_Comm* comm,
                          MPIL_Info* info,
                          MPIL_Request** req_ptr);
						  
						  
#if defined(MPI4)
/**@brief calls underlying PMPI_Alltoallv implementation **/
int alltoallv_init_pmpi(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm,
                   MPIL_Info* info,
                   MPIL_Request** req_ptr);
#endif

#ifdef __cplusplus
}
#endif

#endif
