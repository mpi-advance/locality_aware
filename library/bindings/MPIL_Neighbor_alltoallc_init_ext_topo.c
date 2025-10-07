#include "../../include/neighborhood/neighbor_init.h"

#include "../../include/neighborhood/neighbor.h"
#include "../../include/persistent/neighbor_persistent.h"

enum NeighborAlltoallvInitMethod mpix_neighbor_alltoallv_init_implementation =
    NEIGHBOR_ALLTOALLV_INIT_STANDARD;


int MPIL_Neighbor_alltoallv_init_ext_topo(const void* sendbuf,
                                          const int sendcounts[],
                                          const int sdispls[],
                                          const long global_sindices[],
                                          MPI_Datatype sendtype,
                                          void* recvbuf,
                                          const int recvcounts[],
                                          const int rdispls[],
                                          const long global_rindices[],
                                          MPI_Datatype recvtype,
                                          MPIL_Topo* topo,
                                          MPIL_Comm* comm,
                                          MPIL_Info* info,
                                          MPIL_Request** request_ptr)
{
    switch (mpix_neighbor_alltoallv_init_implementation)
    {
        case NEIGHBOR_ALLTOALLV_INIT_STANDARD:
            return neighbor_alltoallv_init_standard(sendbuf,
                                                    sendcounts,
                                                    sdispls,
                                                    sendtype,
                                                    recvbuf,
                                                    recvcounts,
                                                    rdispls,
                                                    recvtype,
                                                    topo,
                                                    comm,
                                                    info,
                                                    request_ptr);
        case NEIGHBOR_ALLTOALLV_INIT_LOCALITY:
            return neighbor_alltoallv_init_locality_ext(sendbuf,
                                                        sendcounts,
                                                        sdispls,
                                                        global_sindices,
                                                        sendtype,
                                                        recvbuf,
                                                        recvcounts,
                                                        rdispls,
                                                        global_rindices,
                                                        recvtype,
                                                        topo,
                                                        comm,
                                                        info,
                                                        request_ptr);
        default:
            return neighbor_alltoallv_init_standard(sendbuf,
                                                    sendcounts,
                                                    sdispls,
                                                    sendtype,
                                                    recvbuf,
                                                    recvcounts,
                                                    rdispls,
                                                    recvtype,
                                                    topo,
                                                    comm,
                                                    info,
                                                    request_ptr);
    }
}
