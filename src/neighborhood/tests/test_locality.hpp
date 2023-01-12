#ifndef TEST_LOCALITY_HPP
#define TEST_LOCALITY_HPP

#include "mpi.h"
#include "mpi_advance.h"

void update_locality(MPIX_Comm* comm_dist_graph, int ppn)
{
    int rank, num_procs;
    MPI_Comm_rank(comm_dist_graph->global_comm, &rank);
    MPI_Comm_size(comm_dist_graph->global_comm, &num_procs);

    if (comm_dist_graph->local_comm)
        MPI_Comm_free(&(comm_dist_graph->local_comm));

    MPI_Comm_split(comm_dist_graph->global_comm,
            rank / ppn,
            rank,
            &(comm_dist_graph->local_comm));

    MPI_Comm_size(comm_dist_graph->local_comm, &(comm_dist_graph->ppn));
    comm_dist_graph->num_nodes = ((num_procs-1) / comm_dist_graph->ppn) + 1;
    comm_dist_graph->rank_node = get_node(comm_dist_graph, rank);

    int local_rank;
    MPI_Comm_rank(comm_dist_graph->local_comm, &local_rank);

    MPI_Comm_split(comm_dist_graph->global_comm,
            local_rank,
            rank,
            &(comm_dist_graph->group_comm));
}

#endif
