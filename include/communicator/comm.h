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

typedef struct _MPIL_Comm MPIL_Comm;

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



#ifdef __cplusplus
}
#endif

#endif
