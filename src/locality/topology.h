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

    MPI_Win win;
    char* win_array;
    int win_bytes;
    int win_type_bytes;

    MPI_Request* requests;
    int n_requests;
} MPIX_Comm;

int MPIX_Comm_init(MPIX_Comm** xcomm_ptr, MPI_Comm global_comm);
int MPIX_Comm_free(MPIX_Comm* xcomm);

int MPIX_Comm_topo_init(MPIX_Comm* xcomm);
int MPIX_Comm_topo_free(MPIX_Comm* xcomm);

int MPIX_Comm_win_init(MPIX_Comm* xcomm, int bytes, int type_bytes);
int MPIX_Comm_win_free(MPIX_Comm* xcomm);

int MPIX_Comm_req_resize(MPIX_Comm* xcomm, int n);

int get_node(const MPIX_Comm* data, const int proc);
int get_local_proc(const MPIX_Comm* data, const int proc);
int get_global_proc(const MPIX_Comm* data, const int node, const int local_proc);

// For testing purposes (manually set PPN)
void update_locality(MPIX_Comm* xcomm, int ppn);

#ifdef __cplusplus
}
#endif




#endif
