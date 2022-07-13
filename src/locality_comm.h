#ifndef MPI_ADVANCE_LOCALITY_COMM_H
#define MPI_ADVANCE_LOCALITY_COMM_H

#include "comm_pkg.h"

typedef struct _LocalityComm
{
    CommPkg* local_L_comm;
    CommPkg* local_S_comm;
    CommPkg* local_R_comm;
    CommPkg* global_comm;
} LocalityComm;

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


#endif
