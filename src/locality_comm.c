#include "locality_comm.h"

void init_locality_comm(LocalityComm** locality_ptr, MPI_Comm comm)
{
    LocalityComm* locality = (LocalityComm*)malloc(sizeof(LocalityComm));

    init_comm_pkg(&(locality->local_L_comm), 19234);
    init_comm_pkg(&(locality->local_S_comm), 92835);
    init_comm_pkg(&(locality->local_R_comm), 29301);
    init_comm_pkg(&(locality->global_comm), 72459);

    *locality_ptr = locality;
}

void destroy_locality_comm(LocalityComm* locality)
{
    destroy_comm_pkg(locality->local_L_comm);
    destroy_comm_pkg(locality->local_S_comm);
    destroy_comm_pkg(locality->local_R_comm);
    destroy_comm_pkg(locality->global_comm);

    free(locality);
}

