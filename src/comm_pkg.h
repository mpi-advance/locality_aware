#ifndef MPI_ADVANCE_COMM_PKG_H
#define MPI_ADVANCE_COMM_PKG_H

#include "comm_data.h"

typedef struct _CommPkg
{
    CommData* send_data;
    CommData* recv_data;
    int tag;
} CommPkg;

void init_comm_pkg(CommPkg** comm_ptr, int _tag);
void destroy_comm_pkg(CommPkg* comm);

#endif
