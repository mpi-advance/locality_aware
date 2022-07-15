// TODO Currently Assumes SMP Ordering 
//      And equal number of processes per node

#ifndef MPI_ADVANCE_TOPOLOGY_H
#define MPI_ADVANCE_TOPOLOGY_H

#include <mpi.h>
#include "locality_comm.h"

int get_node(const MPIX_Comm* data, const int proc);
int get_local_proc(const MPIX_Comm* data, const int proc);
int get_global_proc(const MPIX_Comm* data, const int node, const int local_proc);


#endif
