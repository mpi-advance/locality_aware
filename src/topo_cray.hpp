//TODO -- this file should be updated to not rely on environment variables
//        but can I determine node from rank, local rank from rank,
//        or global rank from local rank and node, without ordering?
#ifndef NAPCOMM_TOPO_CRAY_HPP
#define NAPCOMM_TOPO_CRAY_HPP

#include <mpi.h>

#define SMP 1
#define RR 0
#define FRR 2


static int get_ordering()
{
    char* proc_layout_c = getenv("MPICH_RANK_REORDER_METHOD");
    if (proc_layout_c)
    {
        return atoi(proc_layout_c);
    }
    else return SMP;
}

static int get_node(const int proc, const int rank_ordering, const int num_nodes, const int ppn)
{
    if (rank_ordering == RR)
    {
        return proc % num_nodes;
    }
    else if (rank_ordering == SMP)
    {
        return proc / ppn;
    }
    else if (rank_ordering == FRR)
    {
        if ((proc / num_nodes) % 2 == 0)
        {
            return proc % num_nodes;
        }
        else
        {
            return num_nodes - (proc % num_nodes) - 1;
        }
    }
    else
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            printf("This MPI rank ordering is not supported!\n");
        }
        return -1;
    }
}

static int get_local_proc(const int proc, const int rank_ordering, const int num_nodes, const int ppn)
{
    if (rank_ordering == RR || rank_ordering == FRR)
    {
        return proc / num_nodes;
    }
    else if (rank_ordering == SMP)
    {
        return proc % ppn;
    }
    else
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            printf("This MPI rank ordering is not supported!\n");
        }
        return -1;
    }
}

static int get_global_proc(const int node, const int local_proc, const int rank_ordering,
        const int num_nodes, const int ppn)
{
    if (rank_ordering == RR)
    {
        return local_proc * num_nodes + node;
    }
    else if (rank_ordering == SMP)
    {
        return local_proc + (node * ppn);
    }
    else if (rank_ordering == FRR)
    {
        if (local_proc % 2 == 0)
        {
            return local_proc * num_nodes + node;
        }
        else
        {
            return local_proc * num_nodes + num_nodes - node - 1;
        }
    }
    else
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
            printf("This MPI rank ordering is not supported!\n");
        }
        return -1;
    }
}

#endif
