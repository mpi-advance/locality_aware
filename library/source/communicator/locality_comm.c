#include "../../include/communicator/locality_comm.h"

void init_locality_comm(LocalityComm** locality_ptr,
                        MPIL_Comm* mpix_comm,
                        MPI_Datatype sendtype,
                        MPI_Datatype recvtype)
{
    LocalityComm* locality = (LocalityComm*)malloc(sizeof(LocalityComm));

    int tag;
    MPIL_Comm_tag(mpix_comm, &tag);
    init_comm_pkg(&(locality->local_L_comm), sendtype, recvtype, tag);

    MPIL_Comm_tag(mpix_comm, &tag);
    init_comm_pkg(&(locality->local_S_comm), sendtype, recvtype, tag);

    MPIL_Comm_tag(mpix_comm, &tag);
    init_comm_pkg(&(locality->local_R_comm), recvtype, recvtype, tag);

    MPIL_Comm_tag(mpix_comm, &tag);
    init_comm_pkg(&(locality->global_comm), recvtype, recvtype, tag);

    locality->communicators = mpix_comm;

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

void get_local_comm_data(LocalityComm* locality,
                         int* max_local_num,
                         int* max_local_size,
                         int* max_non_local_num,
                         int* max_non_local_size)
{
    int sizes[4];
    int max_sizes[4];
    sizes[0] = 0;
    sizes[1] = 0;
    if (locality->local_L_comm)
    {
        sizes[0] += locality->local_L_comm->send_data->num_msgs;
        sizes[1] += locality->local_L_comm->send_data->size_msgs;
    }
    if (locality->local_S_comm)
    {
        sizes[0] += locality->local_S_comm->send_data->num_msgs;
        sizes[1] += locality->local_S_comm->send_data->size_msgs;
    }
    if (locality->local_R_comm)
    {
        sizes[0] += locality->local_R_comm->send_data->num_msgs;
        sizes[1] += locality->local_R_comm->send_data->size_msgs;
    }
    sizes[2] = 0;
    sizes[3] = 0;
    if (locality->global_comm)
    {
        sizes[2] = locality->global_comm->send_data->num_msgs;
        sizes[3] = locality->global_comm->send_data->size_msgs;
    }

    MPI_Allreduce(MPI_IN_PLACE, max_sizes, 4, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    *max_local_num      = sizes[0];
    *max_local_size     = sizes[1];
    *max_non_local_num  = sizes[2];
    *max_non_local_size = sizes[3];
}
