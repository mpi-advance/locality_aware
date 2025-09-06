#include "allreduce.h"

int allreduce_hierarchical(const void *sendbuf, 
                           void* recvbuf,
                           const int count,
                           MPI_Datatype datatype,
                           MPI_Op op,
                           MPIX_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm.global_comm, &rank);
    MPI_Comm_size(comm.global_comm, &num_procs);

    char* send_buffer = (char*) sendbuf;
    char* recv_buffer = (char*) recvbuf;

    int datasize;
    MPI_Type_size(datatype, &datasize);

    if (comm.local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(&comm);

    int local_rank, ppn;
    MPI_Comm_rank(comm.local_comm, &local_rank);
    MPI_Comm_size(comm.local_comm, &ppn);

    int n_nodes = num_procs / ppn;

    char* tmp_buf = NULL;

    if (local_rank == 0)
        tmp_buf = (char*) malloc(count * datasize);
    else
        tmp_buf = (char*) malloc(sizeof(char));

    MPI_Reduce(sendbuf, tmp_buf, count, datatype, op, 0, comm.local_comm);

    if (local_rank == 0)
    {
        MPI_Allreduce(tmp_buf, recv_buffer, count, datatype, op, comm.group_comm);
    }

    MPI_Bcast(recv_buffer, count, datatype, 0, comm.local_comm);

    free(tmp_buf);

    return MPI_SUCCESS;
}

int allreduce_multileader(const void *sendbuf,
                          void *recvbuf,
                          const int count,
                          MPI_Datatype datatype,
                          MPI_Op op,
                          MPIX_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm.leader_comm, &rank);
    MPI_Comm_size(comm.global_comm, &num_procs);

    int tag = 10242;

    if (comm.local_comm == MPI_COMM_NULL)
    {
        MPIX_Comm_topo_init(&comm);
    }

    int num_leaders_per_node = 4;
    int procs_per_node;
    MPI_Comm_size(comm.local_comm, &procs_per_node);
    int procs_per_leader = procs_per_node / num_leaders_per_node;
    if (procs_per_node < num_leaders_per_node)
    {
        num_leaders_per_node = procs_per_node;
        procs_per_leader = 1;
    }

    int send_proc, recv_proc;
    int send_pos, recv_pos;

    char *recv_bufer = (char *)recvbuf;
    char *send_buffer = (char *)sendbuf;

    int data_size;
    MPI_Type_size(datatype, &data_size);

    if (comm.leader_comm == MPI_COMM_NULL)
        MPIX_Comm_leader_init(&comm, procs_per_leader);

    int local_rank, ppn;
    MPI_Comm_rank(comm.leader_comm, &local_rank);
    MPI_Comm_size(comm.leader_comm, &ppn);

    int n_nodes = num_procs / ppn;

    char *local_recv_buff = NULL;
    if (local_rank == 0)
    {
        local_recv_buff = (char *)malloc(count * data_size);
    }
    else
    {
        local_recv_buff = (char *)malloc(sizeof(char));
    }

    // 1. Reduce locally
    MPI_Reduce(sendbuf, local_recv_buff, count, datatype, op, 0, comm.leader_comm);

    if (local_rank == 0)
    {
        // 2. allreduce between leaders
        MPI_Allreduce(local_recv_buff, recvbuf, count, datatype, op, comm.leader_group_comm);
    }

    // 3. Broadcast
    MPI_Bcast(recvbuf, count, datatype, 0, comm.leader_comm);

    free(local_recv_buff);

    return MPI_SUCCESS;
}

int allreduce_node_aware(const void *sendbuf,
                         void *recvbuf,
                         const int count,
                         MPI_Datatype datatype,
                         MPI_Op op,
                         MPIX_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm.global_comm, &rank);
    MPI_Comm_size(comm.global_comm, &num_procs);

    if (comm.local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(&comm);

    int local_rank, ppn;
    MPI_Comm_rank(comm.local_comm, &local_rank);
    MPI_Comm_size(comm.local_comm, &ppn);

    MPI_Status status;
    char* recv_buffer = (char*) recvbuf;
    char* send_buffer = (char*) sendbuf;

    int datasize;
    MPI_Type_size(datatype, &datasize);

    int n_nodes = num_procs / ppn;
    
    char* tmpbuf = (char*) malloc(count * datasize);

    MPI_Allreduce(sendbuf, tmpbuf, count, datatype, op, comm.group_comm);
    MPI_Allreduce(tmpbuf, recvbuf, count, datatype, op, comm.local_comm);

    free(tmpbuf);

    return MPI_SUCCESS;
}

int allreduce_locality_aware(const void *sendbuf,
                             void *recvbuf,
                             const int count,
                             MPI_Datatype datatype,
                             MPI_Op op,
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

    if (comm.leader_comm == MPI_COMM_NULL)
        MPIX_Comm_leader_init(&comm, procs_per_leader);

    int ppn;
    MPI_Comm_size(comm.leader_comm, &ppn);

    int data_size;
    MPI_Type_size(datatype, &data_size);

    int n_nodes = num_procs / ppn;

    char* tmpbuf = (char*) malloc(count * data_size);

    MPI_Allreduce(sendbuf, tmpbuf, count, datatype, op, comm.leader_group_comm);
    MPI_Allreduce(tmpbuf, recvbuf, count, datatype, op, comm.leader_comm);

    free(tmpbuf);

    return MPI_SUCCESS;    
}
