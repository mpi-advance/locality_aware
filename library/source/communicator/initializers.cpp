#include <stdlib.h>

#include "communicator/MPIL_Comm.hpp"

int initialize_comm_object(MPIL_Comm** xcomm_ptr, MPI_Comm global_comm)
{
    MPIL_Comm* xcomm   = (MPIL_Comm*)malloc(sizeof(MPIL_Comm));
    xcomm->global_comm = global_comm;

    new (&xcomm->local_comm) Communicator::CachedComm(MPI_COMM_NULL);
    new (&xcomm->group_comm) Communicator::CachedComm(MPI_COMM_NULL);

    xcomm->leader_comm       = MPI_COMM_NULL;
    xcomm->leader_group_comm = MPI_COMM_NULL;
    xcomm->leader_local_comm = MPI_COMM_NULL;

    xcomm->neighbor_comm = MPI_COMM_NULL;

    xcomm->win       = MPI_WIN_NULL;
    xcomm->win_array = NULL;
    xcomm->win_bytes = 0;

    xcomm->requests   = NULL;
    xcomm->statuses   = NULL;
    xcomm->n_requests = 0;

    int flag;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &(xcomm->max_tag), &flag);
    xcomm->tag = 126 % xcomm->max_tag;

    xcomm->global_rank_to_local = NULL;
    xcomm->global_rank_to_node  = NULL;
    xcomm->ordered_global_ranks = NULL;

#ifdef GPU
    xcomm->gpus_per_node = 0;
#endif

    *xcomm_ptr = xcomm;

    return MPI_SUCCESS;
}

int initialize_rank_mapping(MPIL_Comm* xcomm)
{
    int rank = -1, num_procs = -1;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);
    int local_rank, ppn;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_size(xcomm->local_comm, &ppn);
    int local_node;
    MPI_Comm_rank(xcomm->group_comm, &local_node);

    // Gather arrays for get_node, get_local, and get_global methods
    // These arrays allow for these methods to work with any ordering
    // No longer relying on SMP ordering of processes to nodes!
    // Does rely on constant ppn

    if (xcomm->global_rank_to_local == NULL)
    {
        xcomm->global_rank_to_local = (int*)malloc(num_procs * sizeof(int));
    }

    if (xcomm->global_rank_to_node == NULL)
    {
        xcomm->global_rank_to_node = (int*)malloc(num_procs * sizeof(int));
    }

    MPI_Allgather(&local_rank,
                  1,
                  MPI_INT,
                  xcomm->global_rank_to_local,
                  1,
                  MPI_INT,
                  xcomm->global_comm);
    MPI_Allgather(&local_node,
                  1,
                  MPI_INT,
                  xcomm->global_rank_to_node,
                  1,
                  MPI_INT,
                  xcomm->global_comm);

    if (xcomm->ordered_global_ranks == NULL)
    {
        xcomm->ordered_global_ranks = (int*)malloc(num_procs * sizeof(int));
    }

    for (int i = 0; i < num_procs; i++)
    {
        int local                                       = xcomm->global_rank_to_local[i];
        int node                                        = xcomm->global_rank_to_node[i];
        xcomm->ordered_global_ranks[node * ppn + local] = i;
    }

    // Set xcomm variables
    MPI_Comm_size(xcomm->local_comm, &(xcomm->ppn));
    xcomm->num_nodes = ((num_procs - 1) / xcomm->ppn) + 1;
    xcomm->rank_node = get_node(xcomm, rank);

    return MPI_SUCCESS;
}

int free_rank_mapping(MPIL_Comm* xcomm)
{
    free(xcomm->global_rank_to_local);
    xcomm->global_rank_to_local = NULL;

    free(xcomm->global_rank_to_node);
    xcomm->global_rank_to_node = NULL;

    free(xcomm->ordered_global_ranks);
    xcomm->ordered_global_ranks = NULL;

    return MPI_SUCCESS;
}