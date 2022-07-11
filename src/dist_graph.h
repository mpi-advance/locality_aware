#ifndef MPI_ADVANCE_DIST_GRAPH_H
#define MPI_ADVANCE_DIST_GRAPH_H

#include "mpi.h"
#include "topology.h"

typedef struct _MPIX_Comm
{
    MPI_Comm global_comm;
    MPI_Comm local_comm;
    MPI_Comm neighbor_comm;

    int num_nodes;
    int rank_node;
    int ppn;
} MPIX_Comm;

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

    MPIX_Comm* comm_dist_graph = (MPIX_Comm*)malloc(sizeof(MPIX_Comm));
    comm_dist_graph->global_comm = comm_old;

    MPI_Dist_graph_create_adjacent(comm_dist_graph->global_comm,
            indegree,
            sources,
            outdegree,
            destinations,
            destweights,
            info, 
            reorder,
            &(comm_dist_graph->neighbor_comm));

    MPI_Comm_split_type(comm_dist_graph->global_comm,
            MPI_COMM_TYPE_SHARED,
            rank,
            MPI_INFO_NULL,
            &(comm_dist_graph->local_comm));

    MPI_Comm_size(comm_dist_graph->local_comm, &(comm_dist_graph->ppn));
    comm_dist_graph->num_nodes = ((num_procs-1) / comm_dist_graph->ppn) + 1;
    comm_dist_graph->rank_node = get_node(comm_dist_graph, rank);
    
    *comm_dist_graph_ptr = comm_dist_graph;
}


int MPIX_Comm_free(MPIX_Comm* comm_dist_graph)
{
    MPI_Comm_free(&(comm_dist_graph->neighbor_comm));
    MPI_Comm_free(&(comm_dist_graph->local_comm));

    free(comm_dist_graph);
}

#endif
