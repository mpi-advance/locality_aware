#include "collective.h"

// TODO : currently root is always 0
int bcast(void* buffer,
        int count,
        MPI_Datatype datatype,
        int root,
        MPI_Comm comm);
