#ifndef MPI_ADVANCE_ALLTOALL_H
#define MPI_ADVANCE_ALLTOALL_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "communicator/MPIL_Comm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* // TODO : need to add batch/batch asynch as underlying options for Alltoall
enum AlltoallMethod
{
    ALLTOALL_PAIRWISE,
    ALLTOALL_NONBLOCKING,
    ALLTOALL_HIERARCHICAL_PAIRWISE,
    ALLTOALL_HIERARCHICAL_NONBLOCKING,
    ALLTOALL_MULTILEADER_PAIRWISE,
    ALLTOALL_MULTILEADER_NONBLOCKING,
    ALLTOALL_NODE_AWARE_PAIRWISE,
    ALLTOALL_NODE_AWARE_NONBLOCKING,
    ALLTOALL_LOCALITY_AWARE_PAIRWISE,
    ALLTOALL_LOCALITY_AWARE_NONBLOCKING,
    ALLTOALL_MULTILEADER_LOCALITY_PAIRWISE,
    ALLTOALL_MULTILEADER_LOCALITY_NONBLOCKING,
    ALLTOALL_PMPI
};
extern enum AlltoallMethod mpil_alltoall_implementation;
 */
/** @brief wrapper around other options**/
/* int MPIL_Alltoall(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* mpi_comm); */

typedef int (*alltoall_ftn)(
    const void*, const int, MPI_Datatype, void*, const int, MPI_Datatype, MPIL_Comm*);
typedef int (*alltoall_helper_ftn)(const void*,
                                   const int,
                                   MPI_Datatype,
                                   void*,
                                   const int,
                                   MPI_Datatype,
                                   MPI_Comm,
                                   int tag);

//** External Wrappers **//
/** @brief set tag and call pairwise_helper **/
int alltoall_pairwise(const void* sendbuf,
                      const int sendcount,
                      MPI_Datatype sendtype,
                      void* recvbuf,
                      const int recvcount,
                      MPI_Datatype recvtype,
                      MPIL_Comm* comm);

/** @brief set message tag and call pairwise_helper **/
int alltoall_nonblocking(const void* sendbuf,
                         const int sendcount,
                         MPI_Datatype sendtype,
                         void* recvbuf,
                         const int recvcount,
                         MPI_Datatype recvtype,
                         MPIL_Comm* comm);


/** @brief call alltoall_hiearchical passing pairwise_helper**/
int alltoall_hierarchical_pairwise(const void* sendbuf,
                                   const int sendcount,
                                   MPI_Datatype sendtype,
                                   void* recvbuf,
                                   const int recvcount,
                                   MPI_Datatype recvtype,
                                   MPIL_Comm* comm);

/** @brief call alltoall_hiearchical passing nonblocking_helper**/
int alltoall_hierarchical_nonblocking(const void* sendbuf,
                                      const int sendcount,
                                      MPI_Datatype sendtype,
                                      void* recvbuf,
                                      const int recvcount,
                                      MPI_Datatype recvtype,
                                      MPIL_Comm* comm);

/** @brief call alltoall_hiearchical passing pairwise_helper, nleaders=4**/
int alltoall_multileader_pairwise(const void* sendbuf,
                                  const int sendcount,
                                  MPI_Datatype sendtype,
                                  void* recvbuf,
                                  const int recvcount,
                                  MPI_Datatype recvtype,
                                  MPIL_Comm* comm);
								  
/** @brief call alltoall_hiearchical passing nonblocking_helper, nleaders=4**/							  
int alltoall_multileader_nonblocking(const void* sendbuf,
                                     const int sendcount,
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcount,
                                     MPI_Datatype recvtype,
                                     MPIL_Comm* comm);

/** @brief call node_aware with pairwise helper **/
int alltoall_node_aware_pairwise(const void* sendbuf,
                                 const int sendcount,
                                 MPI_Datatype sendtype,
                                 void* recvbuf,
                                 const int recvcount,
                                 MPI_Datatype recvtype,
                                 MPIL_Comm* comm);

/** @brief call node_aware with nonblocking helper **/
int alltoall_node_aware_nonblocking(const void* sendbuf,
                                    const int sendcount,
                                    MPI_Datatype sendtype,
                                    void* recvbuf,
                                    const int recvcount,
                                    MPI_Datatype recvtype,
                                    MPIL_Comm* comm);

/** @brief call locality_aware with pairwise helper, groups_per_node=4**/
int alltoall_locality_aware_pairwise(const void* sendbuf,
                                     const int sendcount,
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcount,
                                     MPI_Datatype recvtype,
                                     MPIL_Comm* comm);

/** @brief call locality_aware with nonblocking helper, groups_per_node=4**/
int alltoall_locality_aware_nonblocking(const void* sendbuf,
                                        const int sendcount,
                                        MPI_Datatype sendtype,
                                        void* recvbuf,
                                        const int recvcount,
                                        MPI_Datatype recvtype,
                                        MPIL_Comm* comm);

/** @brief calls multileader_locality with pairwise helper **/
int alltoall_multileader_locality_pairwise(const void* sendbuf,
                                           const int sendcount,
                                           MPI_Datatype sendtype,
                                           void* recvbuf,
                                           const int recvcount,
                                           MPI_Datatype recvtype,
                                           MPIL_Comm* comm);

/** @brief calls multileader_locality with nonblocking helper **/
int alltoall_multileader_locality_nonblocking(const void* sendbuf,
                                              const int sendcount,
                                              MPI_Datatype sendtype,
                                              void* recvbuf,
                                              const int recvcount,
                                              MPI_Datatype recvtype,
                                              MPIL_Comm* comm);


//** Intermediate Wrappers **//
/** @brief calls alltoall_locality_aware with groups_per_node=1**/
int alltoall_node_aware(alltoall_helper_ftn f,
                        const void* sendbuf,
                        const int sendcount,
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        const int recvcount,
                        MPI_Datatype recvtype,
                        MPIL_Comm* comm);

/** @brief wrapper around alltoall_multileader, nleaders=1)**/
int alltoall_hierarchical(alltoall_helper_ftn f,
                          const void* sendbuf,
                          const int sendcount,
                          MPI_Datatype sendtype,
                          void* recvbuf,
                          const int recvcount,
                          MPI_Datatype recvtype,
                          MPIL_Comm* comm);



//** Core Helper functions **//
/** @brief Uses Sendrecv to do the alltoall**/
int pairwise_helper(const void* sendbuf,
                    const int sendcount,
                    MPI_Datatype sendtype,
                    void* recvbuf,
                    const int recvcount,
                    MPI_Datatype recvtype,
                    MPI_Comm comm,
                    int tag);

/** @brief Uses Isend and Irecv to do the alltoall**/
int nonblocking_helper(const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPI_Comm comm,
                       int tag);

/** @brief ??? \todo fill**/
int alltoall_multileader(alltoall_helper_ftn f,
                         const void* sendbuf,
                         const int sendcount,
                         MPI_Datatype sendtype,
                         void* recvbuf,
                         const int recvcount,
                         MPI_Datatype recvtype,
                         MPIL_Comm* comm,
                         int n_leaders);

/** @brief complex returns locality_helper **/
int alltoall_locality_aware(alltoall_helper_ftn f,
                            const void* sendbuf,
                            const int sendcount,
                            MPI_Datatype sendtype,
                            void* recvbuf,
                            const int recvcount,
                            MPI_Datatype recvtype,
                            MPIL_Comm* comm,
                            int groups_per_node);


/** @brief ??? \todo fill**/
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

/** @brief ??? \todo fill**/
int alltoall_multileader_locality(alltoall_helper_ftn f,
                                  const void* sendbuf,
                                  const int sendcount,
                                  MPI_Datatype sendtype,
                                  void* recvbuf,
                                  const int recvcount,
                                  MPI_Datatype recvtype,
                                  MPIL_Comm* comm);

// Calls underlying MPI implementation
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
