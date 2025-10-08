#include "../../../../include/communicator/mpil_comm.h"



int MPIL_Comm_leader_init(MPIL_Comm* xcomm, int procs_per_leader)
{
    int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    MPI_Comm_split(
        xcomm->global_comm, rank / procs_per_leader, rank, &(xcomm->leader_comm));

    int leader_rank;
    MPI_Comm_rank(xcomm->leader_comm, &leader_rank);

    MPI_Comm_split(xcomm->global_comm, leader_rank, rank, &(xcomm->leader_group_comm));

    if (xcomm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(xcomm);
    }

    MPI_Comm_split(xcomm->local_comm, leader_rank, rank, &(xcomm->leader_local_comm));

    return MPI_SUCCESS;
}
