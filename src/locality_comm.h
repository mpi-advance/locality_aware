#ifndef MPI_ADVANCE_LOCALITY_COMM_H
#define MPI_ADVANCE_LOCALITY_COMM_H

#include "comm_pkg.h"
#include "dist_graph.h"

typedef struct _LocalityComm
{
    CommPkg* local_L_comm;
    CommPkg* local_S_comm;
    CommPkg* local_R_comm;
    CommPkg* global_comm;
} LocalityComm;

void init_locality_comm(LocalityComm** locality_ptr, MPI_Comm comm);
void destroy_locality_comm(LocalityComm* locality);

#endif
