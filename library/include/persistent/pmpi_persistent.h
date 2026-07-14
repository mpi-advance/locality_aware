#ifndef MPI_ADVANCE_PMPI_PERSISTENT
#define MPI_ADVANCE_PMPI_PERSISTENT

#include "persistent/MPIL_Request.h"

int pmpi_start(MPIL_Request* request);
int pmpi_wait(MPIL_Request* request, MPI_Status* status);

#endif
