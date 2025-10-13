// TODO Currently Assumes SMP Ordering
//      And equal number of processes per node

#ifndef MPI_ADVANCE_TOPOLOGY_H
#define MPI_ADVANCE_TOPOLOGY_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef GPU
#include "../utils/gpu_utils.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _MPIL_Comm
{
    MPI_Comm global_comm;  
	
    MPI_Comm neighbor_comm;

    //For hierarchical collectives
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
    gpuStream_t proc_stream;
#endif
} MPIL_Comm;

int MPIL_Comm_init(MPIL_Comm** xcomm_ptr, MPI_Comm global_comm);
int MPIL_Comm_free(MPIL_Comm** xcomm_ptr);

int MPIL_Comm_topo_init(MPIL_Comm* xcomm);
int MPIL_Comm_topo_free(MPIL_Comm* xcomm);

int MPIL_Comm_leader_init(MPIL_Comm* xcomm, int procs_per_leader);
int MPIL_Comm_leader_free(MPIL_Comm* xcomm);

int MPIL_Comm_win_init(MPIL_Comm* xcomm, int bytes, int type_bytes);
int MPIL_Comm_win_free(MPIL_Comm* xcomm);

int MPIL_Comm_device_init(MPIL_Comm* xcomm);
int MPIL_Comm_device_free(MPIL_Comm* xcomm);

int MPIL_Comm_req_resize(MPIL_Comm* xcomm, int n);

/** @brief get current tag and increment tag in the comm.**/
int MPIL_Comm_tag(MPIL_Comm* comm, int* tag);

int get_node(const MPIL_Comm* data, const int proc);
int get_local_proc(const MPIL_Comm* data, const int proc);
int get_global_proc(const MPIL_Comm* data, const int node, const int local_proc);

// For testing purposes (manually set PPN)
void update_locality(MPIL_Comm* xcomm, int ppn);

#ifdef __cplusplus
}
#endif

#endif
