#include "../../../../include/communicator/mpil_comm.h"
// For testing purposes
// Manually update aggregation size (ppn)
void update_locality(MPIL_Comm* xcomm, int ppn)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    if (xcomm->local_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->local_comm));
    }
    if (xcomm->group_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->group_comm));
    }

    MPI_Comm_split(xcomm->global_comm, rank / ppn, rank, &(xcomm->local_comm));

    int local_rank;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_split(xcomm->global_comm, local_rank, rank, &(xcomm->group_comm));

    int node;
    MPI_Comm_rank(xcomm->group_comm, &node);

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
    MPI_Allgather(
        &node, 1, MPI_INT, xcomm->global_rank_to_node, 1, MPI_INT, xcomm->global_comm);

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

    MPI_Comm_size(xcomm->local_comm, &(xcomm->ppn));
    xcomm->num_nodes = ((num_procs - 1) / xcomm->ppn) + 1;
    xcomm->rank_node = get_node(xcomm, rank);
}
