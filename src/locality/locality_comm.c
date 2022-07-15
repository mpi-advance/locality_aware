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

