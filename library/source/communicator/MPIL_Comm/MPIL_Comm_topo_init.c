#include "communicator/MPIL_Comm.h"


int MPIL_Comm_topo_init(MPIL_Comm* xcomm)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    // Split global comm into local (per node) communicators
    MPI_Comm_split_type(xcomm->global_comm,
                        MPI_COMM_TYPE_SHARED,
                        rank,
                        MPI_INFO_NULL,
                        &(xcomm->local_comm));

    int local_rank, ppn;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_size(xcomm->local_comm, &ppn);

    // Split global comm into group (per local rank) communicators
    MPI_Comm_split(xcomm->global_comm, local_rank, rank, &(xcomm->group_comm));

    int node;
    MPI_Comm_rank(xcomm->group_comm, &node);

    // Gather arrays for get_node, get_local, and get_global methods
    // These arrays allow for these methods to work with any ordering
    // No longer relying on SMP ordering of processes to nodes!
    // Does rely on constant ppn
    xcomm->global_rank_to_local = (int*)malloc(num_procs * sizeof(int));
    xcomm->global_rank_to_node  = (int*)malloc(num_procs * sizeof(int));
    MPI_Allgather(&local_rank,
                  1,
                  MPI_INT,
                  xcomm->global_rank_to_local,
                  1,
                  MPI_INT,
                  xcomm->global_comm);
    MPI_Allgather(
        &node, 1, MPI_INT, xcomm->global_rank_to_node, 1, MPI_INT, xcomm->global_comm);

    xcomm->ordered_global_ranks = (int*)malloc(num_procs * sizeof(int));
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
