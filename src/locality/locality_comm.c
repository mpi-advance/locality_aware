#include "locality_comm.h"

void init_locality_comm(LocalityComm** locality_ptr, const MPIX_Comm* mpix_comm,
        MPI_Datatype sendtype, MPI_Datatype recvtype)
{
    LocalityComm* locality = (LocalityComm*)malloc(sizeof(LocalityComm));

    init_comm_pkg(&(locality->local_L_comm), sendtype, recvtype, 19234);
    init_comm_pkg(&(locality->local_S_comm), sendtype, recvtype, 92835);
    init_comm_pkg(&(locality->local_R_comm), recvtype, recvtype, 29301);
    init_comm_pkg(&(locality->global_comm), recvtype, recvtype, 72459);

    locality->communicators = mpix_comm;

    *locality_ptr = locality;
}

void finalize_locality_comm(LocalityComm* locality)
{
    finalize_comm_pkg(locality->local_L_comm);
    finalize_comm_pkg(locality->local_S_comm);
    finalize_comm_pkg(locality->local_R_comm);
    finalize_comm_pkg(locality->global_comm);
}

void destroy_locality_comm(LocalityComm* locality)
{
    destroy_comm_pkg(locality->local_L_comm);
    destroy_comm_pkg(locality->local_S_comm);
    destroy_comm_pkg(locality->local_R_comm);
    destroy_comm_pkg(locality->global_comm);

    free(locality);
}

int MPIX_Comm_init(MPIX_Comm** comm_dist_graph_ptr, MPI_Comm global_comm)
{
    int rank, num_procs;
    MPI_Comm_rank(global_comm, &rank);
    MPI_Comm_size(global_comm, &num_procs);

    MPIX_Comm* comm_dist_graph = (MPIX_Comm*)malloc(sizeof(MPIX_Comm));
    comm_dist_graph->global_comm = global_comm;

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


