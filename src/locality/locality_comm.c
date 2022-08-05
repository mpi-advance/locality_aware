#include "locality_comm.h"

void init_locality_comm(LocalityComm** locality_ptr, const MPIX_Comm* mpix_comm,
        MPI_Datatype sendtype, MPI_Datatype recvtype)
{
    LocalityComm* locality = (LocalityComm*)malloc(sizeof(LocalityComm));

    init_comm_pkg(&(locality->local_L_comm), sendtype, recvtype, 19234);
    init_comm_pkg(&(locality->local_S_comm), sendtype, recvtype, 92835);
    init_comm_pkg(&(locality->local_R_comm), recvtype, recvtype, 29301);
    init_comm_pkg(&(locality->global_comm), recvtype, recvtype, 72459);

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
    sizes[0] = locality->local_L_comm->send_data->num_msgs
        + locality->local_S_comm->send_data->num_msgs
        + locality->local_R_comm->send_data->num_msgs;
    sizes[1] = locality->local_L_comm->send_data->size_msgs
        + locality->local_S_comm->send_data->size_msgs
        + locality->local_R_comm->send_data->size_msgs;
    sizes[2] = locality->global_comm->send_data->num_msgs;
    sizes[3] = locality->global_comm->send_data->size_msgs;

    MPI_Allreduce(MPI_IN_PLACE, max_sizes, 4, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    *max_local_num = sizes[0];
    *max_local_size = sizes[1];
    *max_non_local_num = sizes[2];
    *max_non_local_size = sizes[3];

}


