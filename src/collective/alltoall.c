#include "alltoall.h"
#include <string.h>
#include <math.h>

#ifdef GPU
#include "heterogeneous/gpu_alltoall.h"
#endif

/**************************************************
 * Locality-Aware Point-to-Point Alltoall
 *  - Aggregates messages locally to reduce 
 *      non-local communciation
 *  - First redistributes on-node so that each
 *      process holds all data for a subset
 *      of other nodes
 *  - Then, performs inter-node communication
 *      during which each process exchanges
 *      data with their assigned subset of nodes
 *  - Finally, redistribute received data
 *      on-node so that each process holds
 *      the correct final data
 *************************************************/
int MPIX_Alltoall(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* mpi_comm)
{    
#ifdef GPU
    gpuMemoryType send_type, recv_type;
    get_mem_types(sendbuf, recvbuf, &send_type, &recv_type);

    if (send_type == gpuMemoryTypeDevice && 
            recv_type == gpuMemoryTypeDevice)
    {
        return copy_to_cpu_alltoall_pairwise(sendbuf,
                sendcount,
                sendtype,
                recvbuf,
                recvcount,
                recvtype,
                mpi_comm);
    }
    else if (send_type == gpuMemoryTypeDevice ||
            recv_type == gpuMemoryTypeDevice)
    {
        return gpu_aware_alltoall_pairwise(sendbuf,
                sendcount,
                sendtype,
                recvbuf,
                recvcount,
                recvtype,
                mpi_comm);
    }
#endif
    return alltoall_pairwise(sendbuf,
        sendcount,
        sendtype,
        recvbuf,
        recvcount,
        recvtype,
        mpi_comm);
}

int alltoall_pairwise(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag = 10238;

    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    // Send to rank + i
    // Recv from rank - i
    for (int i = 0; i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;
        send_pos = send_proc * sendcount * send_size;
        recv_pos = recv_proc * recvcount * recv_size;

        MPI_Sendrecv(send_buffer + send_pos, 
                sendcount, 
                sendtype, 
                send_proc, 
                tag,
                recv_buffer + recv_pos, 
                recvcount, 
                recvtype, 
                recv_proc, 
                tag,
                comm->global_comm, 
                &status);
    }
    return MPI_SUCCESS;
}

int alltoall_nonblocking(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag = 10239;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2*num_procs*sizeof(MPI_Request));

    // Send to rank + i
    // Recv from rank - i
    for (int i = 0; i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;
        send_pos = send_proc * sendcount * send_size;
        recv_pos = recv_proc * recvcount * recv_size;

        MPI_Isend(send_buffer + send_pos,
                sendcount, 
                sendtype, 
                send_proc,
                tag, 
                comm->global_comm,
                &(requests[i]));
        MPI_Irecv(recv_buffer + recv_pos,
                recvcount,
                recvtype,
                recv_proc,
                tag,
                comm->global_comm,
                &(requests[num_procs + i]));
    }

    MPI_Waitall(2*num_procs, requests, MPI_STATUSES_IGNORE);

    free(requests);
    return MPI_SUCCESS;
}


int alltoall_hierarchical(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{

    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag = 10240;

    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    int local_rank, ppn;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &ppn);

    // TODO: currently assuming full nodes, even ppn per node
    //    this is common, so fair assumption for now
    //    likely need to fix before using in something like Trilinos
    int n_nodes = num_procs / ppn;

    char* local_send_buffer = (char*)malloc(ppn*num_procs*sendcount*send_size);
    char* local_recv_buffer = (char*)malloc(ppn*num_procs*recvcount*recv_size);

    // 1. Gather locally
    MPI_Gather(send_buffer, sendcount*num_procs, sendtype, local_recv_buffer, sendcount*num_procs, sendtype,
            0, comm->local_comm);

    // 2. Re-pack for sends
    // Assumes SMP ordering 
    // TODO: allow for other orderings
    int ctr;

    if (local_rank == 0)
    {
        ctr = 0;
        for (int dest_node = 0; dest_node < n_nodes; dest_node++)
        {
            int dest_node_start = dest_node * ppn * sendcount * send_size;
            for (int origin_proc = 0; origin_proc < ppn; origin_proc++)
            {
                int origin_proc_start = origin_proc * num_procs * sendcount * send_size;
                memcpy(&(local_send_buffer[ctr]), &(local_recv_buffer[origin_proc_start + dest_node_start]),
                        ppn * sendcount * send_size);
                ctr += ppn * sendcount * send_size;
            }
        }

        // 3. MPI_Alltoall between leaders
        MPI_Alltoall(local_send_buffer, ppn * ppn * sendcount, sendtype,
                local_recv_buffer, ppn * ppn * recvcount, recvtype, comm->group_comm);

        // 4. Re-pack for local scatter
        ctr = 0;
        for (int dest_proc = 0; dest_proc < ppn; dest_proc++)
        {
            int dest_proc_start = dest_proc * recvcount * recv_size;
            for (int orig_proc = 0; orig_proc < num_procs; orig_proc++)
            {
                int orig_proc_start = orig_proc * ppn * recvcount * recv_size;
                memcpy(&(local_send_buffer[ctr]), &(local_recv_buffer[orig_proc_start + dest_proc_start]),
                        recvcount * recv_size);
                ctr += recvcount * recv_size;

            }
        }
    }

    // 5. Scatter 
    MPI_Scatter(local_send_buffer, recvcount * num_procs, recvtype, recv_buffer, recvcount * num_procs, recvtype,
            0, comm->local_comm);


    free(local_send_buffer);
    free(local_recv_buffer);

    return MPI_SUCCESS;
}


int alltoall_multileader(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{

    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag = 10241;

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    int num_leaders_per_node = 4;
    int procs_per_node;
    MPI_Comm_size(comm->local_comm, &procs_per_node);
    int procs_per_leader = procs_per_node / num_leaders_per_node;
    if (procs_per_node < num_leaders_per_node)
    {
        num_leaders_per_node = procs_per_node;
        procs_per_leader = 1;
    }

    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    if (comm->leader_comm == MPI_COMM_NULL)
        MPIX_Comm_leader_init(comm, procs_per_leader);

    int local_rank, ppn;
    MPI_Comm_rank(comm->leader_comm, &local_rank);
    MPI_Comm_size(comm->leader_comm, &ppn);

    // TODO: currently assuming full nodes, even ppn per node
    //    this is common, so fair assumption for now
    //    likely need to fix before using in something like Trilinos
    int n_nodes = num_procs / ppn;

    char* local_send_buffer = (char*)malloc(ppn*num_procs*sendcount*send_size);
    char* local_recv_buffer = (char*)malloc(ppn*num_procs*recvcount*recv_size);

    // 1. Gather locally
    MPI_Gather(send_buffer, sendcount*num_procs, sendtype, local_recv_buffer, sendcount*num_procs, sendtype,
            0, comm->leader_comm);

    // 2. Re-pack for sends
    // Assumes SMP ordering 
    // TODO: allow for other orderings
    int ctr;

    if (local_rank == 0)
    {
        ctr = 0;
        for (int dest_node = 0; dest_node < n_nodes; dest_node++)
        {
            int dest_node_start = dest_node * ppn * sendcount * send_size;
            for (int origin_proc = 0; origin_proc < ppn; origin_proc++)
            {
                int origin_proc_start = origin_proc * num_procs * sendcount * send_size;
                memcpy(&(local_send_buffer[ctr]), &(local_recv_buffer[origin_proc_start + dest_node_start]),
                        ppn * sendcount * send_size);
                ctr += ppn * sendcount * send_size;
            }
        }

        // 3. MPI_Alltoall between leaders
        MPI_Alltoall(local_send_buffer, ppn * ppn * sendcount, sendtype,
                local_recv_buffer, ppn * ppn * recvcount, recvtype, comm->leader_group_comm);

        // 4. Re-pack for local scatter
        ctr = 0;
        for (int dest_proc = 0; dest_proc < ppn; dest_proc++)
        {
            int dest_proc_start = dest_proc * recvcount * recv_size;
            for (int orig_proc = 0; orig_proc < num_procs; orig_proc++)
            {
                int orig_proc_start = orig_proc * ppn * recvcount * recv_size;
                memcpy(&(local_send_buffer[ctr]), &(local_recv_buffer[orig_proc_start + dest_proc_start]),
                        recvcount * recv_size);
                ctr += recvcount * recv_size;

            }
        }
    }

    // 5. Scatter 
    MPI_Scatter(local_send_buffer, recvcount * num_procs, recvtype, recv_buffer, recvcount*num_procs, recvtype,
            0, comm->leader_comm);


    free(local_send_buffer);
    free(local_recv_buffer);

    return MPI_SUCCESS;
}

int alltoall_node_aware(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag = 10242;

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    int local_rank, ppn;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &ppn);


    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    int n_nodes = num_procs / ppn;

    char* tmpbuf = (char*)malloc(num_procs*sendcount*send_size);

    // 1. Alltoall between group_comms (all data for any process on node)
    MPI_Alltoall(sendbuf, ppn*sendcount, sendtype, recvbuf, ppn*recvcount, recvtype,
            comm->group_comm);

    // 2. Re-pack
    int ctr = 0;
    for (int dest_proc = 0; dest_proc < ppn; dest_proc++)
    {
        int offset = dest_proc * recvcount * recv_size;
        for (int origin = 0; origin < n_nodes; origin++)
        {
            int node_offset = origin * ppn * recvcount * recv_size;
            memcpy(&(tmpbuf[ctr]), &(recvbuf[node_offset + offset]), recvcount*recv_size);
            ctr += recvcount * recv_size;
        }
    }


    // 3. Local alltoall
    MPI_Alltoall(tmpbuf, n_nodes*recvcount, recvtype, 
            recvbuf, n_nodes*recvcount, recvtype, comm->local_comm);

    // 4. Re-order
    ctr = 0;
    for (int node = 0; node < n_nodes; node++)
    {
        int node_offset = node * recvcount * recv_size;
        for (int dest = 0; dest < ppn; dest++)
        {
            int dest_offset = dest * n_nodes * recvcount * recv_size;
            memcpy(&(tmpbuf[ctr]), &(recvbuf[node_offset + dest_offset]), 
                    recvcount * recv_size);
            ctr += recvcount * recv_size;
        }
    }


    memcpy(recvbuf, tmpbuf, num_procs*recvcount*recv_size);

    free(tmpbuf);

    return MPI_SUCCESS;
}


int alltoall_locality_aware(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag = 10243;

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    int num_leaders_per_node = 4;
    int procs_per_node;
    MPI_Comm_size(comm->local_comm, &procs_per_node);
    int procs_per_leader = procs_per_node / num_leaders_per_node;
    if (procs_per_node < num_leaders_per_node)
    {
        num_leaders_per_node = procs_per_node;
        procs_per_leader = 1;
    }

    if (comm->leader_comm == MPI_COMM_NULL)
        MPIX_Comm_leader_init(comm, procs_per_leader);

    int ppn;
    MPI_Comm_size(comm->leader_comm, &ppn);

    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    int n_nodes = num_procs / ppn;

    char* tmpbuf = (char*)malloc(num_procs*sendcount*send_size);

    // 1. Alltoall between group_comms (all data for any process on node)
    MPI_Alltoall(sendbuf, ppn*sendcount, sendtype, recvbuf, ppn*recvcount, recvtype,
            comm->leader_group_comm);

    // 2. Re-pack
    int ctr = 0;
    for (int dest_proc = 0; dest_proc < ppn; dest_proc++)
    {
        int offset = dest_proc * recvcount * recv_size;
        for (int origin = 0; origin < n_nodes; origin++)
        {
            int node_offset = origin * ppn * recvcount * recv_size;
            memcpy(&(tmpbuf[ctr]), &(recvbuf[node_offset + offset]), recvcount*recv_size);
            ctr += recvcount * recv_size;
        }
    }

    // 3. Local alltoall
    MPI_Alltoall(tmpbuf, n_nodes*recvcount, recvtype, 
            recvbuf, n_nodes*recvcount, recvtype, comm->leader_comm);

    // 4. Re-order
    ctr = 0;
    for (int node = 0; node < n_nodes; node++)
    {
        int node_offset = node * recvcount * recv_size;
        for (int dest = 0; dest < ppn; dest++)
        {
            int dest_offset = dest * n_nodes * recvcount * recv_size;
            memcpy(&(tmpbuf[ctr]), &(recvbuf[node_offset + dest_offset]), 
                    recvcount * recv_size);
            ctr += recvcount * recv_size;
        }
    }

    memcpy(recvbuf, tmpbuf, num_procs*recvcount*recv_size);

    free(tmpbuf);
    return MPI_SUCCESS;
}
int alltoall_multileader_locality(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
   int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag = 10241;

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    int num_leaders_per_node = 4;
    int procs_per_node;
    MPI_Comm_size(comm->local_comm, &procs_per_node);
    int procs_per_leader = procs_per_node / num_leaders_per_node;
    if (procs_per_node < num_leaders_per_node)
    {
        num_leaders_per_node = procs_per_node;
        procs_per_leader = 1;
    }

    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    if (comm->leader_comm == MPI_COMM_NULL)
        MPIX_Comm_leader_init(comm, procs_per_leader);

    int leader_rank;
    MPI_Comm_rank(comm->leader_comm, &leader_rank);
    MPI_Comm_size(comm->leader_comm, &procs_per_leader);

    int local_rank, ppn;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &ppn);

    // TODO: currently assuming full nodes, even procs_per_leader per node
    //    this is common, so fair assumption for now
    //    likely need to fix before using in something like Trilinos
    int n_nodes = num_procs / ppn;
    int n_leaders = num_procs / procs_per_leader;
 
    int leaders_per_node;
    MPI_Comm_size(comm->leader_local_comm, &leaders_per_node);


    char* local_send_buffer = NULL;
    char* local_recv_buffer = NULL;
    if (leader_rank == 0)
    {
        local_send_buffer = (char*)malloc(procs_per_leader*num_procs*sendcount*send_size);
        local_recv_buffer = (char*)malloc(procs_per_leader*num_procs*recvcount*recv_size);
    }
    else
    {
        local_send_buffer = (char*)malloc(sizeof(char));
        local_recv_buffer = (char*)malloc(sizeof(char));
    }
    // 1. Gather locally
    MPI_Gather(send_buffer, sendcount*num_procs, sendtype, local_recv_buffer, sendcount*num_procs, sendtype,
            0, comm->leader_comm);


    // 2. Re-pack for sends
    // Assumes SMP ordering 
    // TODO: allow for other orderings
    int ctr;

    if (leader_rank == 0)
    {
        ctr = 0;
        for (int dest_node = 0; dest_node < n_leaders; dest_node++)
        {
            int dest_node_start = dest_node * procs_per_leader * sendcount * send_size;
            for (int origin_proc = 0; origin_proc < procs_per_leader; origin_proc++)
            {
                int origin_proc_start = origin_proc * num_procs * sendcount * send_size;
                memcpy(&(local_send_buffer[ctr]), &(local_recv_buffer[origin_proc_start + dest_node_start]),
                        procs_per_leader * sendcount * send_size);
                ctr += procs_per_leader * sendcount * send_size;
            }
        }

        // 3. MPI_Alltoall between nodes 
        PMPI_Alltoall(local_send_buffer, ppn*procs_per_leader*sendcount, sendtype, 
                local_recv_buffer, ppn*procs_per_leader*recvcount, recvtype, comm->group_comm);

        // Re-Pack for exchange between local leaders
        ctr = 0;
        for (int local_leader = 0; local_leader < leaders_per_node; local_leader++)
        {
            int leader_start = local_leader*procs_per_leader*procs_per_leader*sendcount*send_size;
            for (int dest_node = 0; dest_node < n_nodes; dest_node++)
            {
                int dest_node_start = dest_node*ppn*procs_per_leader*sendcount*send_size;
                memcpy(&(local_send_buffer[ctr]), &(local_recv_buffer[dest_node_start+leader_start]),
                        procs_per_leader*procs_per_leader*sendcount*send_size);
                ctr += procs_per_leader*procs_per_leader*sendcount*send_size;
            }
        }

        PMPI_Alltoall(local_send_buffer, n_nodes*procs_per_leader*procs_per_leader*sendcount, sendtype, 
                local_recv_buffer, n_nodes*procs_per_leader*procs_per_leader*recvcount, recvtype, comm->leader_local_comm);

        ctr = 0;
        for (int dest_proc = 0; dest_proc < procs_per_leader; dest_proc++)
        {
            int dest_proc_start = dest_proc * recvcount * recv_size;
            
            for (int orig_node = 0; orig_node < n_nodes; orig_node++)
            {
                int orig_node_start = orig_node*procs_per_leader*procs_per_leader*recvcount*recv_size;

                for (int orig_leader = 0; orig_leader < leaders_per_node; orig_leader++)
                {
                    int orig_leader_start = orig_leader*num_procs*leaders_per_node*recvcount*recv_size;
                    for (int orig_proc = 0; orig_proc < procs_per_leader; orig_proc++)
                    {
                        int orig_proc_start = orig_proc*procs_per_leader*recvcount*recv_size;
                        int idx = orig_node_start + orig_leader_start + orig_proc_start + dest_proc_start;
                        memcpy(&(local_send_buffer[ctr]), &(local_recv_buffer[idx]), recvcount*recv_size);
                        ctr += recvcount * recv_size;
                    } 
                }
            }
        }
    }


    // 5. Scatter 
    MPI_Scatter(local_send_buffer, recvcount * num_procs, recvtype, recv_buffer, recvcount*num_procs, recvtype,
            0, comm->leader_comm);

    free(local_send_buffer);
    free(local_recv_buffer);


    return MPI_SUCCESS;
}



