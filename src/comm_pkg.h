#ifndef MPI_ADVANCE_COMM_PKG_H
#define MPI_ADVANCE_COMM_PKG_H

#include "comm_data.h"

typedef struct _CommPkg
{
    CommData* send_data;
    CommData* recv_data;
    int tag;
} CommPkg;

void init_comm_pkg(CommPkg** comm_ptr, int _tag)
{
    CommPkg* comm = (CommPkg*)malloc(sizeof(CommPkg));

    init_comm_data(&(comm->send_data));
    init_comm_data(&(comm->recv_data));
    tag = _tag;

    *comm_ptr = comm;
}

void destroy_comm_pkg(CommPkg* comm)
{
    destroy_comm_data(comm->send_data);
    destroy_comm_data(comm->recv_data);

    free(comm);
}

#endif
