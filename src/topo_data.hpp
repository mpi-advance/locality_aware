//TODO -- this file should be updated to not rely on environment variables
//        but can I determine node from rank, local rank from rank,
//        or global rank from local rank and node, without ordering?
#ifndef NAPCOMM_TOPO_DATA_HPP
#define NAPCOMM_TOPO_DATA_HPP

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

// Rank orderings : round robin, SMP, folded round robin
#define RR 0
#define SMP 1
#define FRR 2

struct topo_data{
    int rank_ordering;
    int ppn;
    int num_nodes;
    int rank_node;
    MPI_Comm local_comm;

    topo_data(MPI_Comm mpi_comm)
    {
        int rank, num_procs;
        MPI_Comm_rank(mpi_comm, &rank);
        MPI_Comm_size(mpi_comm, &num_procs);

        MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);

        // Determine processes per node
        MPI_Comm_size(local_comm, &ppn);

        // Determine number of nodes
        // Assuming num_nodes divides num_procs evenly
        num_nodes = ((num_procs-1) / ppn) + 1;

        set_ordering();  // If not Cray or BGQ, assuming SMP style order
        rank_node = get_node(rank);
    }

    ~topo_data()
    {
        MPI_Comm_free(&local_comm);
    }

    // TODO -- currently only supports CRAY
    void set_ordering()
    {
        char* proc_layout_c = getenv("MPICH_RANK_REORDER_METHOD");
        if (proc_layout_c)
        {
            rank_ordering = atoi(proc_layout_c);
        }
        else rank_ordering = SMP;
    }   

    int get_node(const int proc)
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
                return proc % num_nodes;
            else
                return num_nodes - (proc % num_nodes) - 1;
        }
        else
        {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0)
            {
                printf("This MPI rank ordering is not supported!\n");
            }
        }
        return -1;
    }

    int get_local_proc(const int proc)
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

    int get_global_proc(const int node, const int local_proc)
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
};



#endif
