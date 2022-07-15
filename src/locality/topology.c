#include "topology.h"

int get_node(const MPIX_Comm* data, const int proc)
{
    return proc / data->ppn;
}

int get_local_proc(const MPIX_Comm* data, const int proc)
{
    return proc % data->ppn;
}

int get_global_proc(const MPIX_Comm* data, const int node, const int local_proc)
{
    return local_proc + (node * data->ppn);
}

