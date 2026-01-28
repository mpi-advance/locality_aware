#include <stdlib.h>

#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"
#include "numa.h"

int MPIL_Comm_topo_init(MPIL_Comm* xcomm)
{







    int rank, num_procs,numa_node;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);
    if (numa_available() != -1) {
        numa_node = numa_node_of_cpu(sched_getcpu());
    }





    // Split global comm into local (per node) communicators


    enum Split type = split_implementation;



    if(type ==NUMA){

        MPI_Comm node_comm;
        MPI_Comm_split_type(xcomm->global_comm,
                MPI_COMM_TYPE_SHARED,
                rank,
                MPI_INFO_NULL,
                &node_comm);

        int node_rank, ppn;
        MPI_Comm_size(node_comm, &ppn);
        MPI_Comm_rank(node_comm, &node_rank);

        int numa = numa_node;
        MPI_Comm_split(node_comm, numa, node_rank, &(xcomm->local_comm));

        MPI_Comm_free(node_comm);

    }else if(type ==SOCKET){

        MPI_Comm node_comm;
        MPI_Comm_split_type(xcomm->global_comm,
                MPI_COMM_TYPE_SHARED,
                rank,
                MPI_INFO_NULL,
                &node_comm);

        int node_rank, ppn;
        MPI_Comm_size(node_comm, &ppn);
        MPI_Comm_rank(node_comm, &node_rank);
        int numa = numa_node / 4;
        MPI_Comm_split(node_comm, numa, node_rank, &(xcomm->local_comm));

        MPI_Comm_free(node_comm);



    }else if(type ==NODE||true){
        MPI_Comm_split_type(xcomm->global_comm,
                                      MPI_COMM_TYPE_SHARED,
                                      rank,
                                      MPI_INFO_NULL,
                                      &(xcomm->local_comm));
    }





    int local_rank, ppn;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_size(xcomm- >local_comm, &ppn);





    // Split global comm into group (per local rank) communicators
    MPI_Comm_split(xcomm->global_comm, local_rank, rank, &(xcomm->group_comm));

    int node;
    MPI_Comm_rank(xcomm->group_comm, &node);

    // Gather arrays for get_node, get_local, and get_global methods
    // These arrays allow for these methods to work with any ordering
    // No longer relying on SMP ordering of processes to nodes!
    // Does rely on constant ppn
    xcomm->global_rank_to_local = (int*)malloc(num_procs * sizeof(int));
    xcomm->global_rank_to_node  = (int*)malloc(num_procs * sizeof(int));
    MPI_Allgather(&local_rank,
                  1,
                  MPI_INT,
                  xcomm->global_rank_to_local,
                  1,
                  MPI_INT,
                  xcomm->global_comm);
    MPI_Allgather(
        &node, 1, MPI_INT, xcomm->global_rank_to_node, 1, MPI_INT, xcomm->global_comm);

    xcomm->ordered_global_ranks = (int*)malloc(num_procs * sizeof(int));
    for (int i = 0; i < num_procs; i++)
    {
        int local                                       = xcomm->global_rank_to_local[i];
        int node                                        = xcomm->global_rank_to_node[i];
        xcomm->ordered_global_ranks[node * ppn + local] = i;
    }

    // Set xcomm variables
    MPI_Comm_size(xcomm->local_comm, &(xcomm->ppn));
    xcomm->num_nodes = ((num_procs - 1) / xcomm->ppn) + 1;
    xcomm->rank_node = get_node(xcomm, rank);

    return MPI_SUCCESS;
}
