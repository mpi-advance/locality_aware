#ifndef MPI_ADVANCE_H
#define MPI_ADVANCE_H

#include <mpi.h>

/* Objects offered by this header*/
typedef struct _MPIL_Comm MPIL_Comm;
typedef struct _MPIL_Info MPIL_Info;
typedef struct _MPIL_Topo MPIL_Topo;
typedef struct _MPIL_Request MPIL_Request;

typedef int (*mpix_start_ftn)(MPIL_Request* request);
typedef int (*mpix_wait_ftn)(MPIL_Request* request, MPI_Status* status);

#include "collective/collective.h"

#include "communicator/comm.h"
#include "communicator/info.h"

#include "neighborhood/MPIL_Graph.h"
#include "neighborhood/topo.h"
#include "neighborhood/neighbor.h"
#include "neighborhood/init.h"
#include "neighborhood/sparse_coll.h"

#include "persistent/persistent.h"
#include "utils/MPIL_Alloc.h"

#ifdef GPU
#include "utils/gpu_utils.h"
#endif


#endif
