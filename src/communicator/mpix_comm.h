// TODO Currently Assumes SMP Ordering 
//      And equal number of processes per node

#ifndef MPI_ADVANCE_TOPOLOGY_H
#define MPI_ADVANCE_TOPOLOGY_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils/utils.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _MPIX_Comm
{
    MPI_Comm global_comm;

    // For persistent neighborhood collectives
    MPI_Comm neighbor_comm;

    // For hierarchical collectives
    MPI_Comm local_comm;
    MPI_Comm group_comm;

    // For multileader hierarchical collectives
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
} MPIX_Comm;

int MPIX_Comm_init(MPIX_Comm** xcomm_ptr, MPI_Comm global_comm);
int MPIX_Comm_free(MPIX_Comm** xcomm_ptr);

int MPIX_Comm_topo_init(MPIX_Comm* xcomm);
int MPIX_Comm_topo_free(MPIX_Comm* xcomm);

int MPIX_Comm_leader_init(MPIX_Comm* xcomm, int procs_per_leader);
int MPIX_Comm_leader_free(MPIX_Comm* xcomm);

int MPIX_Comm_win_init(MPIX_Comm* xcomm, int bytes, int type_bytes);
int MPIX_Comm_win_free(MPIX_Comm* xcomm);

int MPIX_Comm_device_init(MPIX_Comm* xcomm);
int MPIX_Comm_device_free(MPIX_Comm* xcomm);

int MPIX_Comm_req_resize(MPIX_Comm* xcomm, int n);

int MPIX_Comm_tag(MPIX_Comm* comm, int* tag);

int get_node(const MPIX_Comm* data, const int proc);
int get_local_proc(const MPIX_Comm* data, const int proc);
int get_global_proc(const MPIX_Comm* data, const int node, const int local_proc);

// For testing purposes (manually set PPN)
void update_locality(MPIX_Comm* xcomm, int ppn);

#ifdef __cplusplus
}
#endif




#endif
