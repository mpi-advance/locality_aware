#include "comm_pkg.h"

void init_comm_pkg(CommPkg** comm_ptr, int _tag)
{
    CommPkg* comm = (CommPkg*)malloc(sizeof(CommPkg));

    init_comm_data(&(comm->send_data));
    init_comm_data(&(comm->recv_data));
    comm->tag = _tag;

    *comm_ptr = comm;
}

void destroy_comm_pkg(CommPkg* comm)
{
    destroy_comm_data(comm->send_data);
    destroy_comm_data(comm->recv_data);

    free(comm);
}

