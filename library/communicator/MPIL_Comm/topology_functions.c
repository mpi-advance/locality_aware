#include "../../../include/communicator/mpil_comm.h"

/****  Topology Functions   ****/
int get_node(const MPIL_Comm* data, const int proc)
{
    return data->global_rank_to_node[proc];
}

int get_local_proc(const MPIL_Comm* data, const int proc)
{
    return data->global_rank_to_local[proc];
}

int get_global_proc(const MPIL_Comm* data, const int node, const int local_proc)
{
    return data->ordered_global_ranks[local_proc + (node * data->ppn)];
}

