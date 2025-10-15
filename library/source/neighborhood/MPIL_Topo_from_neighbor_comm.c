#include "locality_aware.h"
#include "communicator/MPIL_Comm.h"
#include "neighborhood/MPIL_Topo.h"
#include <stdlib.h>

int MPIL_Topo_from_neighbor_comm(MPIL_Comm* comm, MPIL_Topo** mpil_topo_ptr)
{
    MPIL_Topo* mpil_topo = (MPIL_Topo*)malloc(sizeof(MPIL_Topo));

    int weighted;
    MPI_Dist_graph_neighbors_count(
        comm->neighbor_comm, &(mpil_topo->indegree), &(mpil_topo->outdegree), &weighted);

    // Create copy of sources/destinations in MPIL_Topo struct
    mpil_topo->sources      = NULL;
    mpil_topo->destinations = NULL;

    if (mpil_topo->indegree)
    {
        mpil_topo->sources = (int*)malloc(mpil_topo->indegree * sizeof(int));
        if (weighted)
        {
            mpil_topo->sourceweights = (int*)malloc(mpil_topo->indegree * sizeof(int));
        }
        else
        {
            mpil_topo->sourceweights = MPI_UNWEIGHTED;
        }
    }

    if (mpil_topo->outdegree)
    {
        mpil_topo->destinations = (int*)malloc(mpil_topo->outdegree * sizeof(int));
        if (weighted)
        {
            mpil_topo->destweights = (int*)malloc(mpil_topo->outdegree * sizeof(int));
        }
        else
        {
            mpil_topo->destweights = MPI_UNWEIGHTED;
        }
    }

    MPI_Dist_graph_neighbors(comm->neighbor_comm,
                             mpil_topo->indegree,
                             mpil_topo->sources,
                             mpil_topo->sourceweights,
                             mpil_topo->outdegree,
                             mpil_topo->destinations,
                             mpil_topo->destweights);

    *mpil_topo_ptr = mpil_topo;

    return MPI_SUCCESS;
}
