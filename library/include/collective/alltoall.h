#ifndef MPI_ADVANCE_ALLTOALL_H
#define MPI_ADVANCE_ALLTOALL_H

#include <mpi.h>
#include <stdlib.h>

#include "communicator/MPIL_Comm.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Function pointer to alltoall implemenation
 * @details
 * Uses the parameters of standard MPI_Alltoall API, except replacing MPI_Comm with
 * MPIL_Comm most of the behavior is derived from internal parameters in MPIL_Comm.
 * MPIL_API alltoall switch statement targets one of these.
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 **/
typedef int (*alltoall_ftn)(
    const void*, const int, MPI_Datatype, void*, const int, MPI_Datatype, MPIL_Comm*);

/** @brief Function pointer to alltoall helper function..
 * @details
 * Uses the parameters of standard MPI_Alltoall API, plus a tag for additional options.
 * usually invoked by a function of type alltoall_ftn.
 *
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] tag unique tag for matching messages.
 **/
typedef int (*alltoall_helper_ftn)(const void*,
                                   const int,
                                   MPI_Datatype,
                                   void*,
                                   const int,
                                   MPI_Datatype,
                                   MPI_Comm,
                                   int tag);

//** External Wrappers
//**//----------------------------------------------------------------------
/** @brief Call the pairwise implementation.
 * @details calls get_tag() then call pairwise_helper() with the same input parameters
 * plus the found tag.
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @return returns value of the pairwise_helper call.
 */
int alltoall_pairwise(const void* sendbuf,
                      const int sendcount,
                      MPI_Datatype sendtype,
                      void* recvbuf,
                      const int recvcount,
                      MPI_Datatype recvtype,
                      MPIL_Comm* comm);

/** @brief Call the non-blocking implemenation.
 * @details calls get_tag then call nonblocking_helper() with the same input parameters
 * plus the found tag.
 *
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @return returns value of the nonblocking_helper call.
 */
int alltoall_nonblocking(const void* sendbuf,
                         const int sendcount,
                         MPI_Datatype sendtype,
                         void* recvbuf,
                         const int recvcount,
                         MPI_Datatype recvtype,
                         MPIL_Comm* comm);

/** @brief call alltoall_hiearchical passing pairwise_helper()
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @return returns value of the nonblocking_helper call.
 **/
int alltoall_hierarchical_pairwise(const void* sendbuf,
                                   const int sendcount,
                                   MPI_Datatype sendtype,
                                   void* recvbuf,
                                   const int recvcount,
                                   MPI_Datatype recvtype,
                                   MPIL_Comm* comm);

/** @brief call alltoall_hiearchical passing nonblocking_helper() **/
int alltoall_hierarchical_nonblocking(const void* sendbuf,
                                      const int sendcount,
                                      MPI_Datatype sendtype,
                                      void* recvbuf,
                                      const int recvcount,
                                      MPI_Datatype recvtype,
                                      MPIL_Comm* comm);

/** @brief call alltoall_hiearchical() passing pairwise_helper(), nleaders=4**/
int alltoall_multileader_pairwise(const void* sendbuf,
                                  const int sendcount,
                                  MPI_Datatype sendtype,
                                  void* recvbuf,
                                  const int recvcount,
                                  MPI_Datatype recvtype,
                                  MPIL_Comm* comm);

/** @brief call alltoall_hiearchical() passing nonblocking_helper(), nleaders=4**/
int alltoall_multileader_nonblocking(const void* sendbuf,
                                     const int sendcount,
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcount,
                                     MPI_Datatype recvtype,
                                     MPIL_Comm* comm);

/** @brief call alltoall_node_node_aware() with pairwise_helper() **/
int alltoall_node_aware_pairwise(const void* sendbuf,
                                 const int sendcount,
                                 MPI_Datatype sendtype,
                                 void* recvbuf,
                                 const int recvcount,
                                 MPI_Datatype recvtype,
                                 MPIL_Comm* comm);

/** @brief call alltoall_node_aware() with nonblocking_helper() **/
int alltoall_node_aware_nonblocking(const void* sendbuf,
                                    const int sendcount,
                                    MPI_Datatype sendtype,
                                    void* recvbuf,
                                    const int recvcount,
                                    MPI_Datatype recvtype,
                                    MPIL_Comm* comm);

/** @brief call alltoall_locality_aware() with pairwise_helper(), groups_per_node=4**/
int alltoall_locality_aware_pairwise(const void* sendbuf,
                                     const int sendcount,
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcount,
                                     MPI_Datatype recvtype,
                                     MPIL_Comm* comm);

/** @brief call alltoall_locality_aware() with nonblocking_helper(), groups_per_node=4**/
int alltoall_locality_aware_nonblocking(const void* sendbuf,
                                        const int sendcount,
                                        MPI_Datatype sendtype,
                                        void* recvbuf,
                                        const int recvcount,
                                        MPI_Datatype recvtype,
                                        MPIL_Comm* comm);

/** @brief calls alltoall_multileader_locality() with pairwise_helper() **/
int alltoall_multileader_locality_pairwise(const void* sendbuf,
                                           const int sendcount,
                                           MPI_Datatype sendtype,
                                           void* recvbuf,
                                           const int recvcount,
                                           MPI_Datatype recvtype,
                                           MPIL_Comm* comm);

/** @brief calls alltoall_multileader_locality() with nonblocking_helper() **/
int alltoall_multileader_locality_nonblocking(const void* sendbuf,
                                              const int sendcount,
                                              MPI_Datatype sendtype,
                                              void* recvbuf,
                                              const int recvcount,
                                              MPI_Datatype recvtype,
                                              MPIL_Comm* comm);

//** Intermediate Wrappers
//**//-----------------------------------------------------------------
/** @brief calls alltoall_locality_aware() with groups_per_node=1**/
int alltoall_node_aware(alltoall_helper_ftn f,
                        const void* sendbuf,
                        const int sendcount,
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        const int recvcount,
                        MPI_Datatype recvtype,
                        MPIL_Comm* comm);

/** @brief wrapper around alltoall_multileader() with nleaders=1**/
int alltoall_hierarchical(alltoall_helper_ftn f,
                          const void* sendbuf,
                          const int sendcount,
                          MPI_Datatype sendtype,
                          void* recvbuf,
                          const int recvcount,
                          MPI_Datatype recvtype,
                          MPIL_Comm* comm);

/** @brief Sets up messaging groups based on topology
 * @param [in] f helper function to do underlying ptp communication.
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] groups_per_node number of groups per node
 * @return value from call to f
 **/
int alltoall_locality_aware(alltoall_helper_ftn f,
                            const void* sendbuf,
                            const int sendcount,
                            MPI_Datatype sendtype,
                            void* recvbuf,
                            const int recvcount,
                            MPI_Datatype recvtype,
                            MPIL_Comm* comm,
                            int groups_per_node);

//** Core Helper functions
//**//------------------------------------------------------------------
/** @brief Uses Sendrecv to do the alltoall
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [tag] tag int flag used for message matching.
 * @return returns MPI_Success
 **/
int pairwise_helper(const void* sendbuf,
                    const int sendcount,
                    MPI_Datatype sendtype,
                    void* recvbuf,
                    const int recvcount,
                    MPI_Datatype recvtype,
                    MPI_Comm comm,
                    int tag);

/** @brief Nonblocking point to point implementation of alltoall.
 * @details
 * Uses Isend and Irecv to do the alltoall
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [tag] tag int flag used for message matching.
 * @return returns MPI_Success
 **/
int nonblocking_helper(const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPI_Comm comm,
                       int tag);

/** @brief Uses n_leaders to aggregate messages and distribute to other processes.
 * @details
 *   Number of leaders controlled by n_leader parameter.
 *   Communication occurs in three stages:
 *   - Each leader gathers from its grouping
 *   - Leaders exchange using supplied helper function.
 *   - Each leader scatters back among its group.
 *
 * @param [in] f pointer to helper function allows additional functionality
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] n_leaders number of leader processes.
 * @return returns MPI_Success
 */
int alltoall_multileader(alltoall_helper_ftn f,
                         const void* sendbuf,
                         const int sendcount,
                         MPI_Datatype sendtype,
                         void* recvbuf,
                         const int recvcount,
                         MPI_Datatype recvtype,
                         MPIL_Comm* comm,
                         int n_leaders);

/** @brief Groups messages based on topology, uses supplied helper function to do ptp
 * communication.
 * @param [in] f helper function to do underlying ptp communication.
 * @param [in] sendbuf buffer containing data to send
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [in] tag tag forwarded to helper function.
 * @return MPI_SUCCESS
 **/
int alltoall_locality_aware_helper(alltoall_helper_ftn f,
                                   const void* sendbuf,
                                   const int sendcount,
                                   MPI_Datatype sendtype,
                                   void* recvbuf,
                                   const int recvcount,
                                   MPI_Datatype recvtype,
                                   MPIL_Comm* comm,
                                   int groups_per_node,
                                   MPI_Comm local_comm,
                                   MPI_Comm group_comm,
                                   int tag);

/** @brief Multileader alltoall algorithm with topology aware grouping.
 * @details
 *  Uses multiple leaders (currently set to 4) to gather messages based on topology
 *  to create communication map.
 *  Actual communications are implemented by the supplied helper function.
 *  Steps:
 *  - Does a node-wide gather
 *  - repack for sends (assumes SMP ordering)
 *  - invoke helper function to perform alltoall between leaders
 *  - repacks if necessary
 *  - invoke helper function to perform alltoall on node
 *  <br><br>
 *  Currently assumes full nodes and equal procs_per_leader per node
 *
 * @param [in] f pointer to helper function allows additional functionality
 * @param [in] sendcount int number of items in sendbuff
 * @param [in] sendtype MPI_Datatype in sendbuff
 * @param [out] recvbuf buffer to receive messages
 * @param [in] recvcount int number of items expected in recvbuff
 * @param [in] recvtype MPI_Datatype in recvbuff
 * @param [in] comm MPIL_Comm used for context
 * @param [tag] tag int flag used for message matching.
 * @return returns MPI_Success
 **/
int alltoall_multileader_locality(alltoall_helper_ftn f,
                                  const void* sendbuf,
                                  const int sendcount,
                                  MPI_Datatype sendtype,
                                  void* recvbuf,
                                  const int recvcount,
                                  MPI_Datatype recvtype,
                                  MPIL_Comm* comm);

/** @brief calls underlying PMPI_Alltoall implementation **/
int alltoall_pmpi(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* comm);

#ifdef __cplusplus
}
#endif

#endif
