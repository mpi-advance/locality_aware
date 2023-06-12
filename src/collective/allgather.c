#include "allgather.h"
#include "gather.h"
#include "bcast.h"
#include <string.h>
#include <math.h>


int MPIX_Allgather(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
#ifdef bruck
    return allgather_bruck(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
#elif p2p
    return allgather_p2p(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
#elif ring
    return allgather_ring(sendbuf, sendcount, sendtype, recvbuf, recvcound, recvtype, comm);
#elif locality_bruck
    return allgather_local_bruck(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
#elif locality_p2p
    return allgather_local_p2p(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
#elif locality_ring
    return allgather_local_ring(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
#endif 
    
    // Default will call standard p2p
    return allgather_p2p(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    
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
    int num_steps = log2((double)(num_procs));
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



int allgather_loc_p2p(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);
    int local_rank, PPN;

    int recv_size;
    MPI_Type_size(recvtype, &recv_size);
    char* recv_buffer = (char*)(recvbuf);
    
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    int num_nodes = comm->num_nodes;
    int local_node = comm->rank_node;

    int local_idx = -1;

    int pos, proc;
    int tag = 923812;
    int local_tag = 728401;
    int start, end;

    int* ppn_msg_sizes = (int*)malloc(PPN*sizeof(int));
    int* ppn_msg_displs = (int*)malloc((PPN+1)*sizeof(int));
    int num_msgs = num_nodes / PPN; // TODO : this includes talking to self
    int extra = num_nodes % PPN;
    ppn_msg_displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        ppn_msg_sizes[i] = num_msgs;
        if (i < extra) ppn_msg_sizes[i]++;
        ppn_msg_displs[i+1] = ppn_msg_displs[i] + ppn_msg_sizes[i];
        if (ppn_msg_displs[i] <= local_node && ppn_msg_displs[i+1] > local_node)
            local_idx = i;
    }
    num_msgs = ppn_msg_displs[local_rank+1] - ppn_msg_displs[local_rank];
    int first_msg = ppn_msg_displs[local_rank];

    MPI_Request* local_requests = (MPI_Request*)malloc(2*PPN*sizeof(MPI_Request));
    MPI_Request* nonlocal_requests = NULL;
    if (num_msgs)
       nonlocal_requests = (MPI_Request*)malloc(2*num_msgs*sizeof(MPI_Request));

    // Local Gather
    // Put at beginning of recvbuf so other data is contiguous
    pos = local_node * PPN * recvcount;
    PMPI_Allgather(sendbuf, sendcount, sendtype,
            &(recv_buffer[pos*recv_size]), recvcount, recvtype, comm->local_comm);

    // Exchange Inter-Node Messages
    // Local rank exchanges data with nodes in list
    start = ppn_msg_displs[local_rank];
    end = ppn_msg_displs[local_rank+1];
    for (int node = start; node < end; node++)
    {
        proc = node*PPN+local_idx;
        int node_pos = node * PPN * recvcount;
        //printf("Rank %d exchanging with %d\n", rank, proc);
        MPI_Isend(&(recv_buffer[pos*recv_size]), recvcount*PPN, recvtype, proc, tag, comm->global_comm, &(nonlocal_requests[node-start])); 
        MPI_Irecv(&(recv_buffer[node_pos*recv_size]), recvcount*PPN, recvtype, proc, tag, comm->global_comm, &(nonlocal_requests[num_msgs+node-start]));
    }
    MPI_Waitall(2*num_msgs, nonlocal_requests, MPI_STATUSES_IGNORE);

    // Redistribute Locally
    for (int i = 0; i < PPN; i++)
    {
        start = ppn_msg_displs[i];
        end = ppn_msg_displs[i+1];
        MPI_Isend(&(recv_buffer[first_msg*PPN*recvcount*recv_size]), num_msgs*PPN*recvcount,
                recvtype, i, local_tag, comm->local_comm, &(local_requests[i]));
        MPI_Irecv(&(recv_buffer[start*PPN*recvcount*recv_size]), (end - start)*PPN*recvcount,
                recvtype, i, local_tag, comm->local_comm, &(local_requests[PPN+i]));
    }
    MPI_Waitall(2*PPN, local_requests, MPI_STATUSES_IGNORE);

    free(ppn_msg_sizes);
    free(ppn_msg_displs);

    free(local_requests);
    if (num_msgs)
        free(nonlocal_requests);

    return 0;
}


int allgather_loc_bruck(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);
    MPI_Comm local_comm = comm->local_comm;

    char* recv_buffer = (char*)(recvbuf);
    int recv_size;
    MPI_Type_size(recvtype, &recv_size);

    int local_rank, PPN;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    int local_node = rank / PPN;
    int num_nodes = num_procs / PPN;

    int tag = 102943;

    // Perform Local Allgather
    allgather_bruck(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm->local_comm);

    // Perform allgather with PPN nodes at once
    // First send to node - (1 to PPN) nodes and recv from node + (1 to PPN) nodes
    // Local rank 0 sends to node-1, local rank 1 sends to node-2, etc
    // Local rank 0 recvs from node+1, local rank 1 recvs from node+2, etc
    int stride, size, dist;
    int send_proc, recv_proc, recv_pos;
    int num_steps = (log2((double)(num_nodes))-1)/log2((double)(PPN)) + 1;

    MPI_Request requests[2];

    stride = PPN;
    for (int i = 0; i < num_steps; i++)
    {
        // bruck : send to 1 away, then 2 away, then 4 away, etc
        size = sendcount * stride;
        dist = local_rank * stride;

        send_proc = rank - dist;
        if (send_proc < 0) send_proc += num_procs;
        recv_proc = rank + dist;
        if (recv_proc >= num_procs) recv_proc -= num_procs;

        recv_pos = size * local_rank;
        if (local_rank)
        {
            MPI_Isend(recv_buffer, size, sendtype, send_proc, tag, comm->global_comm, &(requests[0]));
            MPI_Irecv(&(recv_buffer[recv_pos*recv_size]), size, sendtype, recv_proc, tag, comm->global_comm, &(requests[1]));
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        }


        allgather_bruck(&(recv_buffer[recv_pos*recv_size]), size, recvtype, recvbuf, size, recvtype, comm->local_comm);

        stride *= PPN;
    }


    if (local_node)
        rotate(recv_buffer, 
                (num_nodes-local_node)*PPN*recvcount*recv_size,
                num_procs*recvcount*recv_size);

    return 0;
}


int allgather_ring_overlap(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* requests)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    char* send_buffer = (char*)(sendbuf);
    char* recv_buffer = (char*)(recvbuf);
    int recv_size;
    MPI_Type_size(recvtype, &recv_size);

    int tag = 393054;
    int send_proc = rank - 1;
    int recv_proc = rank + 1;
    if (send_proc < 0) send_proc += num_procs;
    if (recv_proc >= num_procs) recv_proc -= num_procs;
        
    // Copy my data to correct position in recvbuf
    int pos = rank*recvcount;
    for (int i = 0; i < recvcount*recv_size; i++)
    {
        recv_buffer[pos*recv_size+i] = send_buffer[i];
    }
    int next_pos = pos+recvcount;
    if (next_pos >= num_procs*recvcount) next_pos = 0;

    // Communicate single message to left
    int flag = 0;
    for (int i = 1; i < num_procs; i++)
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

        if (!flag) MPI_Testall(2, requests, &flag, MPI_STATUSES_IGNORE);
    }

    return flag;
}


int allgather_loc_ring(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int local_rank, PPN;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &PPN);

    int recv_size;
    MPI_Type_size(recvtype, &recv_size);

    char* send_buffer = (char*)(sendbuf);
    char* recv_buffer = (char*)(recvbuf);

    int local_node = rank / PPN;
    int num_nodes = num_procs / PPN;

    int tag = 393054;

    // Perform allgather with PPN nodes at once
    // First send to node - (1 to PPN) nodes and recv from node + (1 to PPN) nodes
    // Local rank 0 sends to node-1, local rank 1 sends to node-2, etc
    // Local rank 0 recvs from node+1, local rank 1 recvs from node+2, etc
    int send_node = local_node - 1;
    int recv_node = local_node + 1;
    if (send_node < 0) send_node += num_nodes;
    if (recv_node >= num_nodes) recv_node -= num_nodes;

    int send_proc = send_node*PPN + local_rank;
    int recv_proc = recv_node*PPN + local_rank;

    // Copy my data to correct position in recvbuf
    int pos = rank*recvcount;
    int step_size = PPN*recvcount;
    int node_pos = local_node*PPN*recvcount;

    char* tmpbuf0 = (char*)malloc(recvcount*recv_size*sizeof(char));
    char* tmpbuf1 = (char*)malloc(recvcount*recv_size*sizeof(char));
    char* sendbuf_tmp = tmpbuf0;
    char* recvbuf_tmp = tmpbuf1;
    char* tmp_ptr;
    for (int i = 0; i < recvcount*recv_size; i++)
    {
        sendbuf_tmp[i] = send_buffer[i];
    }

    // For each step, send data to corresponding data on send_node
    // Then, allgather locally
    MPI_Request requests[2];
    for (int i = 0; i < num_nodes-1; i++)
    {
        MPI_Irecv(recvbuf_tmp, recvcount, recvtype, recv_proc, tag, comm->global_comm, &(requests[0]));
        MPI_Isend(sendbuf_tmp, recvcount, recvtype, send_proc, tag, comm->global_comm, &(requests[1]));

        // Allgather locally - with ring
        int finished = allgather_ring_overlap(sendbuf_tmp, recvcount, recvtype, 
                &(recv_buffer[node_pos*recv_size]), recvcount, recvtype, comm->local_comm, requests);

        if (!finished) MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        node_pos += step_size;
        if (node_pos >= num_procs*recvcount) node_pos -= num_procs*recvcount;
        pos += step_size;
        if (pos >= num_procs*recvcount) pos -= num_procs*recvcount;

        tmp_ptr = sendbuf_tmp;
        sendbuf_tmp = recvbuf_tmp;
        recvbuf_tmp = tmp_ptr;
    }
    allgather_ring(sendbuf_tmp, recvcount, recvtype, &(recv_buffer[node_pos*recv_size]), recvcount, recvtype, comm->local_comm);    

    free(tmpbuf0);
    free(tmpbuf1);

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
}




