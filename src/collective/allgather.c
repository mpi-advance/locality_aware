#include "allgather.h"
#include <string.h>
#include <math.h>

int allgather_multileader(const void* sendbuf, 
                          int sendcount, 
                          MPI_Datatype sendtype,
                          void *recvbuf, 
                          int recvcount,
                          MPI_Datatype recvtype, 
                          MPIX_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm.global_comm, &rank);
    MPI_Comm_size(comm.global_comm, &num_procs);

    if (comm.local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(&comm);
    
    int num_leaders_per_node = 4;
    int procs_per_node;
    MPI_Comm_size(comm.local_comm, &procs_per_node);
    int procs_per_leader = procs_per_node / num_leaders_per_node;
    if (procs_per_node < num_leaders_per_node)
    {
        num_leaders_per_node = procs_per_node;
        procs_per_leader = 1;
    }

    int sendsize, recvsize;
    MPI_Type_size(sendtype, &sendsize);
    MPI_Type_size(recvtype, &recvsize);

    char* send_buffer = (char*) sendbuf;
    char* recv_buffer = (char*) recvbuf;

    if (comm.leader_comm == MPI_COMM_NULL)
        MPIX_Comm_leader_init(&comm, procs_per_leader);

    int local_rank, ppn;
    MPI_Comm_rank(comm.leader_comm, &local_rank);
    MPI_Comm_size(comm.leader_comm, &ppn);

    int n_nodes = num_procs / ppn;

    char* temp_buffer = NULL;

    if (local_rank == 0)
    {
        temp_buffer = (char*) malloc(procs_per_leader * sendcount * sendsize); 
    }
    else
    {
        temp_buffer = (char*) malloc(sizeof(char));
    }

    MPI_Gather(sendbuf, sendcount, sendtype, temp_buffer, sendcount, sendtype, 0, comm.leader_comm);

    if (local_rank == 0)
    {
        MPI_Allgather(temp_buffer, procs_per_leader * sendcount, sendtype, recv_buffer, procs_per_leader * recvcount, recvtype, comm.leader_group_comm);
    }

    MPI_Bcast(recv_buffer, recvcount * num_procs, recvtype, 0, comm.leader_comm);

    free(temp_buffer);
    return MPI_SUCCESS;
}


int allgather_hierarchical(const void* sendbuf,
                           int sendcount,
                           MPI_Datatype sendtype,
                           void *recvbuf,
                           int recvcount,
                           MPI_Datatype recvtype,
                           MPIX_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm.global_comm, &rank);
    MPI_Comm_size(comm.global_comm, &num_procs);

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    if (comm.local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(&comm);

    int local_rank, ppn;
    MPI_Comm_rank(comm.local_comm, &local_rank);
    MPI_Comm_size(comm.local_comm, &ppn);

    int n_nodes = num_procs / ppn;

    char* temp_buffer = NULL;

    if (local_rank == 0)
    {
        temp_buffer = (char*) malloc(ppn * sendcount * send_size);
    }
    else
    {
        temp_buffer = (char*) malloc(sizeof(char));
    }

    // MPI_Gather(send_buffer, sendcount, sendtype, temp_buffer, sendcount, sendtype, 0, comm.local_comm);

    if (local_rank == 0)
    {
        MPI_Allgather(temp_buffer, ppn * sendcount, sendtype, recvbuf, ppn * recvcount, recvtype, comm.group_comm);
    }


}
        

int allgather_locality_aware(const void* sendbuf,
                             int sendcount,
                             MPI_Datatype sendtype,
                             void *recvbuf,
                             int recvcount,
                             MPI_Datatype recvtype,
                             MPIX_Comm comm)
{
    return MPI_ERR_OP;
}

int allgather_node_aware(const void* sendbuf,
                         int sendcount,
                         MPI_Datatype sendtype,
                         void* recvbuf,
                         int recvcount,
                         MPI_Datatype recvtype,
                         MPIX_Comm comm)
{
    return MPI_ERR_OP;
}

int allgather_multileader_locality_aware(const void* sendbuf,
                                         int sendcount,
                                         MPI_Datatype sendtype,
                                         void* recvbuf,
                                         int recvcount,
                                         MPI_Datatype recvtype,
                                         MPIX_Comm comm)
{
    return MPI_ERR_OP;
}
