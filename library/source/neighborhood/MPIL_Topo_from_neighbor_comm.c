#include "locality_aware.h"
#include "communicator/MPIL_Comm.h"
#include "neighborhood/MPIL_Topo.h"

#include <cstdlib>

int MPIL_Topo_from_neighbor_comm(MPIL_Comm* comm, MPIL_Topo** mpix_topo_ptr)
{
    MPIL_Topo* mpix_topo = (MPIL_Topo*)malloc(sizeof(MPIL_Topo));

    int weighted;
    MPI_Dist_graph_neighbors_count(
        comm->neighbor_comm, &(mpix_topo->indegree), &(mpix_topo->outdegree), &weighted);

    // Create copy of sources/destinations in MPIL_Topo struct
    mpix_topo->sources      = NULL;
    mpix_topo->destinations = NULL;

    if (mpix_topo->indegree)
    {
        mpix_topo->sources = (int*)malloc(mpix_topo->indegree * sizeof(int));
        if (weighted)
        {
            mpix_topo->sourceweights = (int*)malloc(mpix_topo->indegree * sizeof(int));
        }
        else
        {
            mpix_topo->sourceweights = MPI_UNWEIGHTED;
        }
    }

    if (mpix_topo->outdegree)
    {
        mpix_topo->destinations = (int*)malloc(mpix_topo->outdegree * sizeof(int));
        if (weighted)
        {
            mpix_topo->destweights = (int*)malloc(mpix_topo->outdegree * sizeof(int));
        }
        else
        {
            mpix_topo->destweights = MPI_UNWEIGHTED;
        }
    }

    MPI_Dist_graph_neighbors(comm->neighbor_comm,
                             mpix_topo->indegree,
                             mpix_topo->sources,
                             mpix_topo->sourceweights,
                             mpix_topo->outdegree,
                             mpix_topo->destinations,
                             mpix_topo->destweights);

    *mpix_topo_ptr = mpix_topo;

    return MPI_SUCCESS;
}
