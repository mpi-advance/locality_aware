#ifndef MPIL_COMM_H
#define MPIL_COMM_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _MPIL_Comm
{
    MPI_Comm global_comm;

    MPI_Comm neighbor_comm;

    // For hierarchical collectives
    MPI_Comm local_comm;
    MPI_Comm group_comm;

    MPI_Comm leader_comm;
    MPI_Comm leader_group_comm;
    MPI_Comm leader_local_comm;

    int num_nodes;
    int rank_node;
    int ppn;

    MPI_Win win;
    char* win_array;
    int win_bytes;
    int win_type_bytes;

    MPI_Request* requests;
    MPI_Status* statuses;
    int n_requests;

    int tag;
    int max_tag;

    int* global_rank_to_local;
    int* global_rank_to_node;
    int* ordered_global_ranks;

#ifdef GPU

    int gpus_per_node;
    int rank_gpu;
	//actual type is gpuStream_t, changed to void* to assist compiling. 
    void* proc_stream; 
#endif
} MPIL_Comm;

int get_node(const MPIL_Comm* data, const int proc);
int get_local_proc(const MPIL_Comm* data, const int proc);
int get_global_proc(const MPIL_Comm* data, const int node, const int local_proc);

// For testing purposes (manually set PPN)
// void update_locality(MPIL_Comm* xcomm, int ppn);

int MPIL_Comm_tag(MPIL_Comm* xcomm, int* tag);

#ifdef __cplusplus
}
#endif

#endif