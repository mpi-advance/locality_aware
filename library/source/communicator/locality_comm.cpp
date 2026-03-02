#include "communicator/locality_comm.h"

#include <stdlib.h>

#include "locality_aware.h"

void init_locality_comm(LocalityComm** locality_ptr,
                        MPIL_Comm* mpil_comm,
                        MPI_Datatype sendtype,
                        MPI_Datatype recvtype)
{
    LocalityComm* locality = (LocalityComm*)malloc(sizeof(LocalityComm));

    int tag;
    get_tag(mpil_comm, &tag);
    init_comm_pkg(&(locality->local_L_comm), sendtype, recvtype, tag);

    get_tag(mpil_comm, &tag);
    init_comm_pkg(&(locality->local_S_comm), sendtype, recvtype, tag);

    get_tag(mpil_comm, &tag);
    init_comm_pkg(&(locality->local_R_comm), recvtype, recvtype, tag);

    get_tag(mpil_comm, &tag);
    init_comm_pkg(&(locality->global_comm), recvtype, recvtype, tag);

    locality->communicators = mpil_comm;

    *locality_ptr = locality;
}

void finalize_locality_comm(LocalityComm* locality)
{
    finalize_comm_pkg(locality->local_L_comm);
    finalize_comm_pkg(locality->local_S_comm);
    finalize_comm_pkg(locality->local_R_comm);
    finalize_comm_pkg(locality->global_comm);
}

void destroy_locality_comm(LocalityComm* locality)
{
    destroy_comm_pkg(locality->local_L_comm);
    destroy_comm_pkg(locality->local_S_comm);
    destroy_comm_pkg(locality->local_R_comm);
    destroy_comm_pkg(locality->global_comm);

    free(locality);
}
