#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"

int MPIL_Comm_leader_init(MPIL_Comm* xcomm, int procs_per_leader)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    if (xcomm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(xcomm);
    }
    int local_rank;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);

    int leader = local_rank / procs_per_leader;
    int leader_rank = local_rank % procs_per_leader;

    MPI_Comm_split(
        xcomm->local_comm, local_rank / procs_per_leader, rank, &(xcomm->leader_comm));

    MPI_Comm_split(xcomm->global_comm, leader_rank, rank, &(xcomm->leader_group_comm));
    MPI_Comm_split(xcomm->local_comm, leader_rank, rank, &(xcomm->leader_local_comm));

    return MPI_SUCCESS;
}
