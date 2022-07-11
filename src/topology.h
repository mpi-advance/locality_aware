// TODO Currently Assumes SMP Ordering 
//      And equal number of processes per node

#ifndef MPI_ADVANCE_TOPOLOGY_HPP
#define MPI_ADVANCE_TOPOLOGY_HPP

#include <mpi.h>

int get_node(MPIX_Comm* data, const int proc)
{
    return proc / data->ppn;
}

int get_local_proc(MPIX_Comm* data, const int proc)
{
    return proc % data->ppn;
}

int get_global_proc(MPIX_Comm* data, const int node, const int local_proc)
{
    return local_proc + (node * data->ppn);
}


#endif
