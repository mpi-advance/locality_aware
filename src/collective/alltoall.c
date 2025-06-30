#include "alltoall.h"
#include <string.h>
#include <math.h>

#ifdef GPU
#include "heterogeneous/gpu_alltoall.h"
#endif


// Default alltoall is pairwise
AlltoallMethod mpix_alltoall_implementation = ALLTOALL_PAIRWISE;

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
#ifdef GPU_AWARE
    return gpu_aware_alltoall(alltoall_pairwise,
                sendbuf,
                sendcount,
                sendtype,
                recvbuf,
                recvcount,
                recvtype,
                mpi_comm);
#endif 
#endif
    alltoall_ftn method;

    switch (mpix_alltoall_implementation)
    {
        case ALLTOALL_PAIRWISE:
            method = alltoall_pairwise;
            break;
        case ALLTOALL_NONBLOCKING:
            method = alltoall_nonblocking;
            break;
        case ALLTOALL_HIERARCHICAL_PAIRWISE:
            method = alltoall_hierarchical_pairwise;
            break;
        case ALLTOALL_HIERARCHICAL_NONBLOCKING:
            method = alltoall_hierarchical_nonblocking;
            break;
        case ALLTOALL_MULTILEADER_PAIRWISE:
            method = alltoall_multileader_pairwise;    
            break;
        case ALLTOALL_MULTILEADER_NONBLOCKING:
            method = alltoall_multileader_nonblocking;
            break;
        case ALLTOALL_NODE_AWARE_PAIRWISE:
            method = alltoall_node_aware_pairwise;
            break;
        case ALLTOALL_NODE_AWARE_NONBLOCKING:
            method = alltoall_node_aware_nonblocking;
            break;
        case ALLTOALL_LOCALITY_AWARE_PAIRWISE:
            method = alltoall_locality_aware_pairwise;
            break;
        case ALLTOALL_LOCALITY_AWARE_NONBLOCKING:
            method = alltoall_locality_aware_nonblocking;
            break;
        case ALLTOALL_MULTILEADER_LOCALITY_PAIRWISE:
            method = alltoall_multileader_locality_pairwise;
            break;
        case ALLTOALL_MULTILEADER_LOCALITY_NONBLOCKING:
            method = alltoall_multileader_locality_nonblocking;
            break;
        case ALLTOALL_PMPI:
            method = alltoall_pmpi;
            break;
        default:
            method = alltoall_pmpi;
            break;
    }


    return method(sendbuf,
            sendcount,
            sendtype,
            recvbuf,
            recvcount,
            recvtype,
            mpi_comm);
    
}


int pairwise_helper(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm,
        int tag)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

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
                comm, 
                &status);
    }
    return MPI_SUCCESS;
}

int alltoall_pairwise(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int tag;
    MPIX_Comm_tag(comm, &tag);

    return pairwise_helper(sendbuf, sendcount, sendtype, 
            recvbuf, recvcount, recvtype, comm->global_comm, tag);
}

int nonblocking_helper(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm,
        int tag)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int send_proc, recv_proc;
    int send_pos, recv_pos;

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
                comm,
                &(requests[i]));
        MPI_Irecv(recv_buffer + recv_pos,
                recvcount,
                recvtype,
                recv_proc,
                tag,
                comm,
                &(requests[num_procs + i]));
    }

    MPI_Waitall(2*num_procs, requests, MPI_STATUSES_IGNORE);

    free(requests);
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
    int tag;
    MPIX_Comm_tag(comm, &tag);

    return nonblocking_helper(sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm->global_comm, tag);
}


int alltoall_multileader(
        alltoall_helper_ftn f,
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        int n_leaders)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag;
    MPIX_Comm_tag(comm, &tag);

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    int ppn;
    MPI_Comm_size(comm->local_comm, &ppn);

    MPI_Comm local_comm = comm->local_comm;
    MPI_Comm group_comm = comm->group_comm;


    if (n_leaders > 1)
    {
        if (ppn < n_leaders)
        {
            n_leaders = ppn;
        }
        int procs_per_leader = ppn / n_leaders;

        // If leader comm exists but with wrong number of leaders per node,
        // free the stale communicator
        if (comm->leader_comm != MPI_COMM_NULL)
        {
            int ppl;
            MPI_Comm_size(comm->leader_comm, &ppl);
            if (ppl != procs_per_leader)
                MPI_Comm_free(&comm->leader_comm);
        }

        // If leader comm does not exist, create it
        if (comm->leader_comm == MPI_COMM_NULL)
            MPIX_Comm_leader_init(comm, procs_per_leader);

        local_comm = comm->leader_comm;
        group_comm = comm->leader_group_comm;
    }

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    int local_rank, ppl;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &ppl);

    // TODO: currently assuming full nodes, even ppn per node
    //    this is common, so fair assumption for now
    //    likely need to fix before using in something like Trilinos
    int n_nodes = num_procs / ppl;

    char* local_send_buffer = NULL;
    char* local_recv_buffer = NULL;

    if (local_rank == 0)
    {
        local_send_buffer = (char*)malloc(ppl*num_procs*sendcount*send_size);
        local_recv_buffer = (char*)malloc(ppl*num_procs*recvcount*recv_size);
    }
    else
    {
        local_send_buffer = (char*)malloc(sizeof(char));
        local_recv_buffer = (char*)malloc(sizeof(char));
    }

    // 1. Gather locally
    MPI_Gather(send_buffer, sendcount*num_procs, sendtype, local_recv_buffer, sendcount*num_procs, sendtype,
            0, local_comm);

    // 2. Re-pack for sends
    // Assumes SMP ordering 
    // TODO: allow for other orderings
    int ctr;

    if (local_rank == 0)
    {
        ctr = 0;
        for (int dest_node = 0; dest_node < n_nodes; dest_node++)
        {
            int dest_node_start = dest_node * ppl * sendcount * send_size;
            for (int origin_proc = 0; origin_proc < ppl; origin_proc++)
            {
                int origin_proc_start = origin_proc * num_procs * sendcount * send_size;
                memcpy(&(local_send_buffer[ctr]), &(local_recv_buffer[origin_proc_start + dest_node_start]),
                        ppl * sendcount * send_size);
                ctr += ppl * sendcount * send_size;
            }
        }

        // 3. MPI_Alltoall between leaders
        f(local_send_buffer, ppl * ppl * sendcount, sendtype,
                local_recv_buffer, ppl * ppl * recvcount, recvtype, group_comm, tag);

        // 4. Re-pack for local scatter
        ctr = 0;
        for (int dest_proc = 0; dest_proc < ppl; dest_proc++)
        {
            int dest_proc_start = dest_proc * recvcount * recv_size;
            for (int orig_proc = 0; orig_proc < num_procs; orig_proc++)
            {
                int orig_proc_start = orig_proc * ppl * recvcount * recv_size;
                memcpy(&(local_send_buffer[ctr]), &(local_recv_buffer[orig_proc_start + dest_proc_start]),
                        recvcount * recv_size);
                ctr += recvcount * recv_size;

            }
        }
    }

    // 5. Scatter 
    MPI_Scatter(local_send_buffer, recvcount * num_procs, recvtype, recv_buffer, recvcount*num_procs, recvtype,
            0, local_comm);


    free(local_send_buffer);
    free(local_recv_buffer);

    return MPI_SUCCESS;
}

int alltoall_hierarchical(
        alltoall_helper_ftn f,
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return alltoall_multileader(f, sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm, 1);
}

int alltoall_hierarchical_pairwise(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return alltoall_hierarchical(pairwise_helper, sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm);
}

int alltoall_hierarchical_nonblocking(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return alltoall_hierarchical(nonblocking_helper, sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm);
}

int alltoall_multileader_pairwise(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return alltoall_multileader(pairwise_helper, sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm, 4);
}

int alltoall_multileader_nonblocking(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return alltoall_multileader(nonblocking_helper, sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm, 4);
}



int alltoall_locality_aware_helper(
        alltoall_helper_ftn f,
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        int groups_per_node,
        MPI_Comm local_comm,
        MPI_Comm group_comm,
        int tag)
{ 
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int ppg;
    MPI_Comm_size(local_comm, &ppg);

    char* recv_buffer = (char*)recvbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    int n_groups = num_procs / ppg;

    char* tmpbuf = (char*)malloc(num_procs*sendcount*send_size);

    // 1. Alltoall between group_comms (all data for any process on node)
    f(sendbuf, ppg*sendcount, sendtype, tmpbuf, ppg*recvcount, recvtype,
            group_comm, tag);

    // 2. Re-pack
    int ctr = 0;
    for (int dest_proc = 0; dest_proc < ppg; dest_proc++)
    {
        int offset = dest_proc * recvcount * recv_size;
        for (int origin = 0; origin < n_groups; origin++)
        {
            int node_offset = origin * ppg * recvcount * recv_size;
            memcpy(&(recv_buffer[ctr]), &(tmpbuf[node_offset + offset]), recvcount*recv_size);
            ctr += recvcount * recv_size;
        }
    }

    // 3. Local alltoall
    f(recvbuf, n_groups*recvcount, recvtype, 
            tmpbuf, n_groups*recvcount, recvtype, local_comm, tag);

    // 4. Re-order
    ctr = 0;
    for (int node = 0; node < n_groups; node++)
    {
        int node_offset = node * recvcount * recv_size;
        for (int dest = 0; dest < ppg; dest++)
        {
            int dest_offset = dest * n_groups * recvcount * recv_size;
            memcpy(&(recv_buffer[ctr]), &(tmpbuf[node_offset + dest_offset]), 
                    recvcount * recv_size);
            ctr += recvcount * recv_size;
        }
    }

    free(tmpbuf);
    return MPI_SUCCESS;
}

int alltoall_locality_aware(
        alltoall_helper_ftn f,
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        int groups_per_node)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag;
    MPIX_Comm_tag(comm, &tag);

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    int ppn;
    MPI_Comm_size(comm->local_comm, &ppn);

    MPI_Comm local_comm = comm->local_comm;
    MPI_Comm group_comm = comm->group_comm;

    if (groups_per_node > 1)
    {
        if (ppn < groups_per_node)
        {
            groups_per_node = ppn;
        }
        int procs_per_group = ppn / groups_per_node;

        if (comm->leader_comm != MPI_COMM_NULL)
        {
            int ppg;
            MPI_Comm_size(comm->leader_comm, &ppg);
            if (ppg != procs_per_group)
                MPI_Comm_free(&(comm->leader_comm));
        }

        if (comm->leader_comm == MPI_COMM_NULL)
            MPIX_Comm_leader_init(comm, procs_per_group);

        local_comm = comm->leader_comm;
        group_comm = comm->leader_group_comm;
    }

    return alltoall_locality_aware_helper(f, sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm, groups_per_node, local_comm, group_comm, tag);
}

int alltoall_node_aware(
        alltoall_helper_ftn f,
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return alltoall_locality_aware(f, sendbuf, sendcount, sendtype, 
            recvbuf, recvcount, recvtype, comm, 1);
}

int alltoall_node_aware_pairwise(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return alltoall_node_aware(pairwise_helper, sendbuf, sendcount, sendtype, 
            recvbuf, recvcount, recvtype, comm);
}

int alltoall_node_aware_nonblocking(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return alltoall_node_aware(nonblocking_helper, sendbuf, sendcount, sendtype, 
            recvbuf, recvcount, recvtype, comm);
}

int alltoall_locality_aware_pairwise(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return alltoall_locality_aware(pairwise_helper, sendbuf, sendcount, sendtype, 
            recvbuf, recvcount, recvtype, comm, 4);
}

int alltoall_locality_aware_nonblocking(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return alltoall_locality_aware(nonblocking_helper, sendbuf, sendcount, sendtype, 
            recvbuf, recvcount, recvtype, comm, 4);
}




int alltoall_multileader_locality(
        alltoall_helper_ftn f,
        const void* sendbuf,
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

    int tag;
    MPIX_Comm_tag(comm, &tag);

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    int local_rank, ppn;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &ppn);

    if (comm->leader_comm == MPI_COMM_NULL)
    {
        int num_leaders_per_node = 4;
        if (ppn < num_leaders_per_node)
            num_leaders_per_node = ppn;
        MPIX_Comm_leader_init(comm, ppn / num_leaders_per_node);
    }
    
    int procs_per_leader, leader_rank;
    MPI_Comm_rank(comm->leader_comm, &leader_rank);
    MPI_Comm_size(comm->leader_comm, &procs_per_leader);

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

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
    /*
        alltoall_locality_aware_helper(f, sendbuf, procs_per_leader*sendcount, sendtype,
            recvbuf, procs_per_leader*recvcount, recvtype, comm, groups_per_node, 
            comm->leader_local_comm, comm->group_comm);
*/

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
        f(local_send_buffer, ppn*procs_per_leader*sendcount, sendtype, 
                local_recv_buffer, ppn*procs_per_leader*recvcount, recvtype, comm->group_comm, tag);

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

        f(local_send_buffer, n_nodes*procs_per_leader*procs_per_leader*sendcount, sendtype, 
                local_recv_buffer, n_nodes*procs_per_leader*procs_per_leader*recvcount, recvtype, 
                comm->leader_local_comm, tag);

        ctr = 0;
        for (int dest_proc = 0; dest_proc < procs_per_leader; dest_proc++)
        {
            int dest_proc_start = dest_proc * recvcount * recv_size;
            
            for (int orig_node = 0; orig_node < n_nodes; orig_node++)
            {
                int orig_node_start = orig_node*procs_per_leader*procs_per_leader*recvcount*recv_size;

                for (int orig_leader = 0; orig_leader < leaders_per_node; orig_leader++)
                {
                    int orig_leader_start = orig_leader*n_nodes*procs_per_leader*procs_per_leader*recvcount*recv_size;
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


int alltoall_multileader_locality_pairwise(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return alltoall_multileader_locality(pairwise_helper,
            sendbuf, sendcount, sendtype, recvbuf, recvcount, 
            recvtype, comm);
}

int alltoall_multileader_locality_nonblocking(
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return alltoall_multileader_locality(nonblocking_helper,
            sendbuf, sendcount, sendtype, recvbuf, recvcount, 
            recvtype, comm);
}



// Calls underlying MPI implementation
int alltoall_pmpi(const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount,
            recvtype, comm->global_comm);
}


