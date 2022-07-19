#include "dist_graph.h"

int MPIX_Dist_graph_create_adjacent(MPI_Comm comm_old, 
        int indegree,
        const int sources[],
        const int sourceweights[],
        int outdegree,
        const int destinations[],
        const int destweights[],
        MPI_Info info,
        int reorder,
        MPIX_Comm** comm_dist_graph_ptr)
{
    int rank, num_procs;
    MPI_Comm_rank(comm_old, &rank);
    MPI_Comm_size(comm_old, &num_procs);

    MPIX_Comm* comm_dist_graph;
    MPIX_Comm_init(&comm_dist_graph, comm_old);

    MPI_Dist_graph_create_adjacent(comm_dist_graph->global_comm,
            indegree,
            sources,
            sourceweights,
            outdegree,
            destinations,
            destweights,
            info, 
            reorder,
            &(comm_dist_graph->neighbor_comm));

    *comm_dist_graph_ptr = comm_dist_graph;
}



