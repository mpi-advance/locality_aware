#include "topology.h"

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

    int local_rank, ppn;
    MPI_Comm_rank(comm_dist_graph->local_comm, &local_rank);
    MPI_Comm_size(comm_dist_graph->local_comm, &ppn);

    MPI_Comm_split(comm_dist_graph->global_comm,
            local_rank,
            rank,
            &(comm_dist_graph->group_comm));

    comm_dist_graph->neighbor_comm = MPI_COMM_NULL;
    
    *comm_dist_graph_ptr = comm_dist_graph;

#ifdef GPU
    gpuGetDeviceCount(&(comm_dist_graph->gpus_per_node));
    comm_dist_graph->ranks_per_gpu = ppn / comm_dist_graph->gpus_per_node;
    comm_dist_graph->rank_gpu = rank / comm_dist_graph->ranks_per_gpu;
    comm_dist_graph->gpu_rank = rank % comm_dist_graph->ranks_per_gpu;
    gpuStreamCreate(&(comm_dist_graph->proc_stream));    

    MPI_Comm_split(comm_dist_graph->local_comm,
            comm_dist_graph->rank_gpu,
            local_rank,
            &(comm_dist_graph->gpu_comm));

    MPI_Comm_split(comm_dist_graph->global_comm,
            comm_dist_graph->gpu_rank,
            rank,
            &(comm_dist_graph->gpu_group_comm));

#endif


    return 0;
}

int MPIX_Comm_free(MPIX_Comm* comm_dist_graph)
{
    if (comm_dist_graph->neighbor_comm != MPI_COMM_NULL)
        MPI_Comm_free(&(comm_dist_graph->neighbor_comm));
    MPI_Comm_free(&(comm_dist_graph->local_comm));
    MPI_Comm_free(&(comm_dist_graph->group_comm));

#ifdef GPU
    gpuStreamDestroy(comm_dist_graph->proc_stream);
    MPI_Comm_free(&(comm_dist_graph->gpu_comm));
    MPI_Comm_free(&(comm_dist_graph->gpu_group_comm));
#endif

    free(comm_dist_graph);

    return 0;
}

int get_node(const MPIX_Comm* data, const int proc)
{
    return proc / data->ppn;
}

int get_local_proc(const MPIX_Comm* data, const int proc)
{
    return proc % data->ppn;
}

int get_global_proc(const MPIX_Comm* data, const int node, const int local_proc)
{
    return local_proc + (node * data->ppn);
}

// For testing purposes
// Manually update aggregation size (ppn)
void update_locality(MPIX_Comm* comm_dist_graph, int ppn)
{
    int rank, num_procs;
    MPI_Comm_rank(comm_dist_graph->global_comm, &rank);
    MPI_Comm_size(comm_dist_graph->global_comm, &num_procs);

    if (comm_dist_graph->local_comm != MPI_COMM_NULL)
        MPI_Comm_free(&(comm_dist_graph->local_comm));
    if (comm_dist_graph->group_comm != MPI_COMM_NULL)
        MPI_Comm_free(&(comm_dist_graph->group_comm));

    MPI_Comm_split(comm_dist_graph->global_comm,
        rank / ppn,
        rank,
        &(comm_dist_graph->local_comm));

    MPI_Comm_size(comm_dist_graph->local_comm, &(comm_dist_graph->ppn));
    comm_dist_graph->num_nodes = ((num_procs - 1) / comm_dist_graph->ppn) + 1;
    comm_dist_graph->rank_node = get_node(comm_dist_graph, rank);

    int local_rank;
    MPI_Comm_rank(comm_dist_graph->local_comm, &local_rank);
    MPI_Comm_split(comm_dist_graph->global_comm,
        local_rank,
        rank,
        &(comm_dist_graph->group_comm));
}

