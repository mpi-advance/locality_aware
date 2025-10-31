#ifndef MPI_ADVANCE_ALLTOALLV_H
#define MPI_ADVANCE_ALLTOALLV_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "communicator/MPIL_Comm.h"

#ifdef __cplusplus
extern "C" {
#endif
/** @brief function pointer to alltoallv implemenation
 * @details 
 * Uses the parameters of standard MPI_Alltoallv API, except replacing MPI_Comm with MPIL_Comm
 * most of the behavior is derived from internal parameters in MPIL_Comm.
 * MPIL_API alltoallv switch statement targets one of these.  
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to recieve messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 **/
typedef int (*alltoallv_ftn)(const void*,
                             const int*,
                             const int*,
                             MPI_Datatype,
                             void*,
                             const int*,
                             const int*,
                             MPI_Datatype,
                             MPIL_Comm*);
							 
/** @brief Uses Sendrecv to do the alltoallv
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to recieve messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 **/
int alltoallv_pairwise(const void* sendbuf,
                       const int sendcounts[],
                       const int sdispls[],
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcounts[],
                       const int rdispls[],
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm);

/** @brief Uses Isend and Irecv to do the alltoallv
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to recieve messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 **/
int alltoallv_nonblocking(const void* sendbuf,
                          const int sendcounts[],
                          const int sdispls[],
                          MPI_Datatype sendtype,
                          void* recvbuf,
                          const int recvcounts[],
                          const int rdispls[],
                          MPI_Datatype recvtype,
                          MPIL_Comm* comm);
						  
/** @brief Uses nonblocking to do the alltoallv operation
 * @details
 *    Has internal tuning parameter nb_stride, which controls the number
 *    of messages between waits. 
 *    
 *    Fires off nb_stride messages before waiting on completion.  
 *
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to recieve messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 **/
int alltoallv_batch(const void* sendbuf,
                    const int sendcounts[],
                    const int sdispls[],
                    MPI_Datatype sendtype,
                    void* recvbuf,
                    const int recvcounts[],
                    const int rdispls[],
                    MPI_Datatype recvtype,
                    MPIL_Comm* comm);
	

/** @brief Uses nonblocking to do the alltoallv operation
 * @details
 *    Has internal tuning parameter nb_stride, which controls the number
 *    of messages between waits. 
 *    
 *    Fires off nb_stride messages then rotates and fires new messages as requests complete.   
 *
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to recieve messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 **/	
int alltoallv_batch_async(const void* sendbuf,
                          const int sendcounts[],
                          const int sdispls[],
                          MPI_Datatype sendtype,
                          void* recvbuf,
                          const int recvcounts[],
                          const int rdispls[],
                          MPI_Datatype recvtype,
                          MPIL_Comm* comm);
						  
/**@brief calls underlying PMPI_Alltoallv implementation **/
int alltoallv_pmpi(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm);

#ifdef __cplusplus
}
#endif

#endif
