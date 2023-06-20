#include "allgather.h"
#include "gather.h"
#include "bcast.h"
#include <string.h>
#include <math.h>
#include "utils.h"


int MPIX_Allgather(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
#ifdef bruck
    return allgather_bruck(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm->global_comm);
#elif p2p
    return allgather_p2p(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm->global_comm);
#elif ring
    return allgather_ring(sendbuf, sendcount, sendtype, recvbuf, recvcound, recvtype, comm->global_comm);
#endif 
    
    // Default will call standard p2p
    return allgather_p2p(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm->global_comm);
    
}





int allgather_bruck(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int tag = 102943;
    MPI_Request requests[2];
    
    char* recv_buffer = (char*)recvbuf;

    int recv_size;
    MPI_Type_size(recvtype, &recv_size);

    // Copy my data to beginning of recvbuf
    if (sendbuf != recvbuf)
        memcpy(recvbuf, sendbuf, recvcount*recv_size);

    // Perform allgather
    int stride;
    int send_proc, recv_proc, size;
    int num_steps = log2(num_procs);
    int msg_size = recvcount*recv_size;

    stride = 1;
    for (int i = 0; i < num_steps; i++)
    {
        send_proc = rank - stride;
        if (send_proc < 0) send_proc += num_procs;
        recv_proc = rank + stride;
        if (recv_proc >= num_procs) recv_proc -= num_procs;
        size = stride*recvcount;

        MPI_Isend(recv_buffer, size, recvtype, send_proc, tag, comm, &(requests[0]));
        MPI_Irecv(recv_buffer + size*recv_size, size, recvtype, recv_proc, tag, comm, &(requests[1]));
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        stride *= 2;
    }

    // Rotate Final Data
    if (rank)
        rotate(recv_buffer, (num_procs-rank)*msg_size, num_procs*msg_size);

    return 0;
}


int allgather_p2p(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    char* recv_buffer = (char*)recvbuf;
    int recv_size;
    MPI_Type_size(recvtype, &recv_size);

    int tag = 204932;
    MPI_Request* requests = (MPI_Request*)malloc(2*num_procs*sizeof(MPI_Request));

    for (int i = 0; i < num_procs; i++)
    {
        MPI_Irecv(&(recv_buffer[i*recvcount*recv_size]), 
                recvcount, recvtype, i, tag, comm, &(requests[i]));
        MPI_Isend(sendbuf, sendcount, sendtype, i, tag, comm, &(requests[num_procs+i]));
    }

    MPI_Waitall(2*num_procs, requests, MPI_STATUSES_IGNORE);
    free(requests);

    return 0;
}

int allgather_ring(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    char* send_buffer = (char*)(sendbuf);
    char* recv_buffer = (char*)(recvbuf);

    int tag = 393054;
    int send_proc = rank - 1;
    int recv_proc = rank + 1;
    if (send_proc < 0) send_proc += num_procs;
    if (recv_proc >= num_procs) recv_proc -= num_procs;

    int recv_size;
    MPI_Type_size(recvtype, &recv_size);
        
    // Copy my data to correct position in recvbuf
    int pos = rank*recvcount;
    for (int i = 0; i < recvcount*recv_size; i++)
    {
        recv_buffer[pos*recv_size+i] = send_buffer[i];
    }
    int next_pos = pos+recvcount;
    if (next_pos >= num_procs*recvcount) next_pos = 0;

    // Communicate single message to left
    for (int i = 0; i < num_procs - 1; i++)
    {
        if (rank % 2)
        {
            MPI_Send(&(recv_buffer[pos*recv_size]), sendcount, sendtype, send_proc, tag, comm);
            MPI_Recv(&(recv_buffer[next_pos*recv_size]), recvcount, recvtype, recv_proc, tag, comm, MPI_STATUS_IGNORE);
        }
        else
        {
            MPI_Recv(&(recv_buffer[next_pos*recv_size]), recvcount, recvtype, recv_proc, tag, comm, MPI_STATUS_IGNORE);
            MPI_Send(&(recv_buffer[pos*recv_size]), sendcount, sendtype, send_proc, tag, comm);
        }
        pos = next_pos;
        next_pos += recvcount;
        if (next_pos >= num_procs*recvcount) next_pos = 0;
    }

    return 0;
}


int allgather_hier_bruck(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPIX_Comm* comm)
{
    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);
    
    int recv_size;
    MPI_Type_size(recvtype, &recv_size);

    int local_rank, PPN;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    char* tmpbuf = (char*)malloc(recvcount*num_procs*recv_size*sizeof(char));

    gather(sendbuf, sendcount, sendtype, tmpbuf, recvcount, recvtype, 0, comm->local_comm);
    if (local_rank == 0)
    {
        allgather_bruck(tmpbuf, recvcount*PPN, recvtype, recvbuf, recvcount*PPN, recvtype, comm->group_comm);
    }
    bcast(recvbuf, recvcount*num_procs, recvtype, 0, comm->local_comm);

    free(tmpbuf);

    return MPI_SUCCESS;
}




int allgather_mult_hier_bruck(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPIX_Comm* comm)
{
    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int recv_size;
    MPI_Type_size(recvtype, &recv_size);

    int group_size;
    MPI_Comm_size(comm->group_comm, &group_size);

    int ppn;
    MPI_Comm_size(comm->local_comm, &ppn);

    char* tmpbuf = (char*)malloc(recvcount*num_procs*recv_size);
    char* recv_buffer = (char*)recvbuf;

    allgather_bruck(sendbuf, sendcount, sendtype, tmpbuf, recvcount, recvtype, comm->group_comm);
    allgather_bruck(tmpbuf, recvcount*group_size, recvtype, 
                tmpbuf, recvcount*group_size, recvtype, comm->local_comm);

    for (int i = 0; i < ppn; i++)
    {
        for (int j = 0; j < group_size; j++)
        {
            for (int k = 0; k < recvcount; k++)
            {
                for (int l = 0; l < recv_size; l++)
                {
                    recv_buffer[j*ppn*recvcount*recv_size + i*recvcount*recv_size + k*recv_size + l]
                        = tmpbuf[i*group_size*recvcount*recv_size + j*recvcount*recv_size + k*recv_size + l];
                }
            }
        }
    }

    free(tmpbuf);

    return MPI_SUCCESS;
}




