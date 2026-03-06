#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"

int MPIL_Comm_leader_init(MPIL_Comm* xcomm, int n_leaders)
{
    int rank;
    MPI_Comm_rank(xcomm->global_comm, &rank);

    if (xcomm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(xcomm);
    }
    int ppn, local_rank;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_size(xcomm->local_comm, &ppn);

    int procs_per_leader = ppn / n_leaders;

    int leader = local_rank / procs_per_leader;
    int leader_rank = local_rank % procs_per_leader;

    MPI_Comm_split(
        xcomm->local_comm, local_rank / procs_per_leader, rank, &(xcomm->leader_comm));

    MPI_Comm_split(xcomm->global_comm, leader_rank, rank, &(xcomm->leader_group_comm));
    MPI_Comm_split(xcomm->local_comm, leader_rank, rank, &(xcomm->leader_local_comm));

    return MPI_SUCCESS;
}
