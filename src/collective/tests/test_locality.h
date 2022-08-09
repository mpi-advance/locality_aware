#ifndef MPI_ADVANCE_TEST_LOCALITY
#define MPI_ADVANCE_TEST_LOCALITY

void allocate_locality(MPIX_Comm** comm_dist_graph_ptr, MPI_Comm global_comm, int ppn)
{
    int rank, num_procs;
    MPI_Comm_rank(global_comm, &rank);
    MPI_Comm_size(global_comm, &num_procs);

    MPIX_Comm* comm_dist_graph = (MPIX_Comm*)malloc(sizeof(MPIX_Comm));
    comm_dist_graph->global_comm = global_comm;

    MPI_Comm_split(comm_dist_graph->global_comm,
            rank/ppn,
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

    comm_dist_graph->neighbor_comm = MPI_COMM_NULL;

    *comm_dist_graph_ptr = comm_dist_graph;

}

#endif
