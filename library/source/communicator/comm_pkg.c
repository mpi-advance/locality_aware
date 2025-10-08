#include "../../../include/communicator/comm_pkg.h"

void init_comm_pkg(CommPkg** comm_ptr,
                   MPI_Datatype sendtype,
                   MPI_Datatype recvtype,
                   int _tag)
{
    CommPkg* comm = (CommPkg*)malloc(sizeof(CommPkg));

    init_comm_data(&(comm->send_data), sendtype);
    init_comm_data(&(comm->recv_data), recvtype);
    comm->tag = _tag;

    *comm_ptr = comm;
}

void finalize_comm_pkg(CommPkg* comm_pkg)
{
    finalize_comm_data(comm_pkg->send_data);
    finalize_comm_data(comm_pkg->recv_data);
}

void destroy_comm_pkg(CommPkg* comm)
{
    destroy_comm_data(comm->send_data);
    destroy_comm_data(comm->recv_data);

    free(comm);
}
