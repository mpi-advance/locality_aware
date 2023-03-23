// TODO Currently Assumes SMP Ordering 
//      And equal number of processes per node

#ifndef MPI_ADVANCE_TOPOLOGY_H
#define MPI_ADVANCE_TOPOLOGY_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _MPIX_Comm
{
    MPI_Comm global_comm;
    MPI_Comm local_comm;
    MPI_Comm neighbor_comm;
    MPI_Comm group_comm;

    int num_nodes;
    int rank_node;
    int ppn;

#ifdef GPU
   int gpus_per_node;
   int ranks_per_gpu;
   int rank_gpu; // GPU my rank corresponds to
   int gpu_rank; // Rank out of all processes associated with rank_gpu
   gpuStream_t proc_stream;

   MPI_Comm gpu_comm; // All processes on node with same rank_gpu
   MPI_Comm gpu_group_comm; // All processes with same gpu_rank
#endif
   
} MPIX_Comm;

int MPIX_Comm_init(MPIX_Comm** comm_dist_graph_ptr, MPI_Comm global_comm);
int MPIX_Comm_free(MPIX_Comm* comm_dist_graph);

int get_node(const MPIX_Comm* data, const int proc);
int get_local_proc(const MPIX_Comm* data, const int proc);
int get_global_proc(const MPIX_Comm* data, const int node, const int local_proc);

// For testing purposes (manually set PPN)
void update_locality(MPIX_Comm* comm_dist_graph, int ppn);

#ifdef __cplusplus
}
#endif




#endif
