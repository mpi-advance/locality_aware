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
    else
        MPI_Comm_size(comm.leader_comm, &procs_per_leader);

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

    MPI_Gather(sendbuf, sendcount, sendtype, temp_buffer, sendcount, sendtype, 0, comm.local_comm);

    if (local_rank == 0)
    {
        MPI_Allgather(temp_buffer, ppn * sendcount, sendtype, recvbuf, ppn * recvcount, recvtype, comm.group_comm);
    }

    MPI_Bcast(recvbuf, recvcount * num_procs, recvtype, 0, comm.local_comm);

    free(temp_buffer);
    return MPI_SUCCESS;
}
        

int allgather_locality_aware(const void* sendbuf,
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

    if (comm.leader_comm == MPI_COMM_NULL)
        MPIX_Comm_leader_init(&comm, procs_per_leader);

    int ppn;
    MPI_Comm_size(comm.leader_comm, &ppn);

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    int n_nodes = num_procs / ppn;

    int ppl;
    MPI_Comm_size(comm.leader_group_comm, &ppl);
    int numLeaders = num_procs / ppl;

    char* tmpbuf = (char*) malloc(ppl * sendcount * send_size);
    char* tmpRecvBuf = (char*) malloc(num_procs * recvcount * recv_size);

    char* recv_buffer = (char*) recvbuf;
    // 1. Allgather between group_comms
    MPI_Allgather(sendbuf, sendcount, sendtype, tmpbuf, recvcount, recvtype, comm.leader_group_comm);

    // 2. Local allgather
    MPI_Allgather(tmpbuf, ppl * recvcount, recvtype, tmpRecvBuf, ppl * recvcount, recvtype, comm.leader_comm);

    // 3. Re-order
    int ctr = 0;
    for (int node = 0; node < n_nodes; node++)
    {
        int node_offset = node * recvcount * recv_size;
        for (int dest = 0; dest < ppn; dest++)
        {
            int dest_offset = dest * n_nodes * recvcount * recv_size;
            memcpy(&(recv_buffer[ctr]), &(tmpRecvBuf[node_offset + dest_offset]), recvcount * recv_size);
            ctr += recvcount * recv_size;
        } 
    }


    free(tmpbuf);
    free(tmpRecvBuf);

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
    int rank, num_procs;
    MPI_Comm_rank(comm.global_comm, &rank);
    MPI_Comm_size(comm.global_comm, &num_procs);

    if (comm.local_comm == MPI_COMM_NULL)   
        MPIX_Comm_topo_init(&comm);

    int local_rank, ppn;
    MPI_Comm_rank(comm.local_comm, &local_rank);
    MPI_Comm_size(comm.local_comm, &ppn);

    int groupCommSize;
    MPI_Comm_size(comm.group_comm, &groupCommSize);

    char* recv_buffer = (char*) recvbuf;
    char* send_buffer = (char*) sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    int n_nodes = num_procs / ppn;

    char* tmpbuf = (char*) malloc(groupCommSize * sendcount * send_size);
    char* tmpRecvBuf = (char*) malloc(num_procs * recvcount * recv_size);

    MPI_Allgather(sendbuf, sendcount, sendtype, tmpbuf, sendcount, sendtype, comm.group_comm);

    MPI_Allgather(tmpbuf, groupCommSize * sendcount, sendtype, tmpRecvBuf, groupCommSize * recvcount, recvtype, comm.local_comm);

    // 3. Re-order
    int ctr = 0;
    for (int node = 0; node < n_nodes; node++)
    {
        int node_offset = node * recvcount * recv_size;
        for (int dest = 0; dest < ppn; dest++)
        {
            int dest_offset = dest * n_nodes * recvcount * recv_size;
            memcpy(&(recv_buffer[ctr]), &(tmpRecvBuf[node_offset + dest_offset]), recvcount * recv_size);
            ctr += recvcount * recv_size;
        } 
    }

    free(tmpbuf);
    free(tmpRecvBuf);

    return MPI_SUCCESS;
}

int allgather_multileader_locality_aware(const void* sendbuf,
                                         int sendcount,
                                         MPI_Datatype sendtype,
                                         void* recvbuf,
                                         int recvcount,
                                         MPI_Datatype recvtype,
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

    if (comm.leader_comm == MPI_COMM_NULL)
    {
        int num_leaders_per_node = 4;
        if (ppn < num_leaders_per_node)
            num_leaders_per_node = ppn;
        MPIX_Comm_leader_init(&comm, ppn / num_leaders_per_node);
    }

    int procs_per_leader, leader_rank;
    MPI_Comm_rank(comm.leader_comm, &leader_rank);
    MPI_Comm_size(comm.leader_comm, &procs_per_leader);

    char* recv_buffer = (char*) recvbuf;
    char* send_buffer = (char*) sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    int n_nodes = num_procs / ppn;
    int n_leaders = num_procs / procs_per_leader;

    int leaders_per_node;
    MPI_Comm_size(comm.leader_local_comm, &leaders_per_node);

    char* local_send_buffer = NULL;
    char* local_recv_buffer = NULL;
    if (leader_rank == 0)
    {
        local_send_buffer = (char*) malloc(procs_per_leader * sendcount * send_size);
        local_recv_buffer = (char*) malloc(num_procs * recvcount * recv_size);
    }
    else
    {
        local_send_buffer = (char*) malloc(sizeof(char));
        local_recv_buffer = (char*) malloc(sizeof(char));
    }

    // 1. Gather to leaders
    MPI_Gather(send_buffer, sendcount, sendtype, local_send_buffer, sendcount, sendtype, 0, comm.leader_comm);

    // 2. Allgather between leaders
    if (leader_rank == 0)
    {
        MPI_Allgather(local_send_buffer, procs_per_leader * sendcount, sendtype, recv_buffer, procs_per_leader * recvcount, recvtype, comm.group_comm);

        MPI_Allgather(recv_buffer, procs_per_leader * leaders_per_node * sendcount, sendtype, local_recv_buffer, procs_per_leader * leaders_per_node * recvcount, recvtype, comm.leader_local_comm); 

        int ctr = 0;
        if (leader_rank == 0)
        {
            for (int leader = 0; leader < leaders_per_node; leader++)
            {
                int leader_offset = leader * procs_per_leader * recvcount * recv_size;
                for (int node = 0; node < n_nodes; node++)
                {
                    int node_offset = node * ppn * recvcount * recv_size;
                    memcpy(&(recv_buffer[ctr]), &(local_recv_buffer[leader_offset + node_offset]), procs_per_leader * recvcount * recv_size);
                    ctr += procs_per_leader * recvcount * recv_size;
                } 
            }
        }
    }

    // 3. Broadcast 
    MPI_Bcast(recvbuf, recvcount * num_procs, recvtype, 0, comm.local_comm);

    free(local_send_buffer);
    free(local_recv_buffer);
    return MPI_SUCCESS;
}
