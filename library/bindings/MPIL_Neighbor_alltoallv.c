#include "../../include/neighborhood/neighbor.h"

#ifdef __cplusplus
#include <cstring>
#endif

#ifndef __cplusplus
#include "string.h"
#endif

int MPIL_Neighbor_alltoallv(const void* sendbuffer,
                            const int sendcounts[],
                            const int sdispls[],
                            MPI_Datatype sendtype,
                            void* recvbuffer,
                            const int recvcounts[],
                            const int rdispls[],
                            MPI_Datatype recvtype,
                            MPIL_Comm* comm)
{
    MPIL_Topo* topo;
    MPIL_Topo_from_neighbor_comm(comm, &topo);

    MPIL_Neighbor_alltoallv_topo(sendbuffer,
                                 sendcounts,
                                 sdispls,
                                 sendtype,
                                 recvbuffer,
                                 recvcounts,
                                 rdispls,
                                 recvtype,
                                 topo,
                                 comm);

    MPIL_Topo_free(&topo);

    return MPI_SUCCESS;
}