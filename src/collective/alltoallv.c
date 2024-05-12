#include "alltoallv.h"
#include <string.h>
#include <math.h>
#include "utils.h"

/**************************************************
 * Locality-Aware Point-to-Point Alltoallv
 * Same as PMPI_Alltoall (no load balancing)
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
 *  - To be used when sizes are relatively balanced
 *  - For load balacing, use persistent version
 *      - Load balacing is too expensive for 
 *          non-persistent Alltoallv
 *************************************************/
int MPI_Alltoallv(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    return alltoallv_pairwise(
        sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf,
        recvcounts,
        rdispls,
        recvtype,
        comm);
}

int MPIX_Alltoallv(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* mpi_comm)
{
    return alltoallv_waitany(sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf,
        recvcounts,
        rdispls,
        recvtype,
        mpi_comm->global_comm);
}


int alltoallv_pairwise(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int tag = 103044;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    memcpy(
        recvbuf + (rdispls[rank] * recv_size),
        sendbuf + (sdispls[rank] * send_size), 
        sendcounts[rank] * send_size);        

    // Send to rank + i
    // Recv from rank - i
    for (int i = 1; i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;

        send_pos = sdispls[send_proc] * send_size;
        recv_pos = rdispls[recv_proc] * recv_size;

        MPI_Sendrecv(sendbuf + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                recvbuf + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                comm, &status);
    }

    return 0;
}

int alltoallv_nonblocking(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int tag = 103044;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2*(num_procs-1)*sizeof(MPI_Request));

    memcpy(
        recvbuf + (rdispls[rank] * recv_size),
        sendbuf + (sdispls[rank] * send_size), 
        sendcounts[rank] * send_size);        

    // For each step i
    // exchange among procs stride (i+1) apart
    for (int i = 1; i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;

        send_pos = sdispls[send_proc] * send_size;
        recv_pos = rdispls[recv_proc] * recv_size;

        MPI_Isend(sendbuf + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                comm, &(requests[i-1]));
        MPI_Irecv(recvbuf + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                comm, &(requests[num_procs+i-2]));
    }

    MPI_Waitall(2*(num_procs-1), requests, MPI_STATUSES_IGNORE);

    free(requests);

    return 0;
}

/**new
 *To be remamed alltoallv_init just like alltoall_init *
 * **/

int alltoallv_nonblocking_init(const void* sendbuf,
       const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
       const int recvcounts[],
       const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* xcomm,
        MPIX_Info* xinfo,
        MPIX_Request** request_ptr)
{
    

 
int rank, num_procs;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);

    MPIX_Request* request;
    MPIX_Request_init(&request);
    request->global_n_msgs = 2*num_procs;
    allocate_requests(request->global_n_msgs, &(request->global_requests));

    request->start_function = batch_start;
    request->wait_function = batch_wait;

    int tag = 102944;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    char* send_buffer = (char*)(sendbuf);
    char* recv_buffer = (char*)(recvbuf);    

    // Initialize persistent send and receive requests
    for (int i = 0; i < num_procs; i++) {
        send_proc = (rank + i) % num_procs;  
        recv_proc = (rank - i + num_procs) % num_procs;  

        send_pos = sdispls[send_proc] * send_size;
        recv_pos = rdispls[recv_proc] * recv_size;


MPI_Send_init(send_buffer + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                xcomm->global_comm, &(request->global_requests[2*i]));
        // Initialize persistent receive request 
        MPI_Recv_init(recvbuf + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                xcomm->global_comm, &(request->global_requests[2*i + 1]));
    }

    // Set the request pointer to the newly created request array
    *request_ptr = request;

    return 0;  // Return success
}





int alltoallv_pairwise_nonblocking(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    // Tuning Parameter : number of non-blocking messages between waits 
    int nb_stride = 5;

    int tag = 103044;
    int ctr;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2*nb_stride*sizeof(MPI_Request));

    memcpy(
        recvbuf + (rdispls[rank] * recv_size),
        sendbuf + (sdispls[rank] * send_size), 
        sendcounts[rank] * send_size);        

    // For each step i
    // exchange among procs stride (i+1) apart
    ctr = 0;
    for (int i = 1; i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;

        send_pos = sdispls[send_proc] * send_size;
        recv_pos = rdispls[recv_proc] * recv_size;

        MPI_Isend(sendbuf + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                comm, &(requests[ctr++]));
        MPI_Irecv(recvbuf + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                comm, &(requests[ctr++]));

        if (i % nb_stride == 0)
        {
            MPI_Waitall(2*nb_stride, requests, MPI_STATUSES_IGNORE);
            ctr = 0;
        }
    }
    
    if (ctr)
        MPI_Waitall(ctr, requests, MPI_STATUSES_IGNORE);

    free(requests);

    return 0;
}

int alltoallv_waitany(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    // Tuning Parameter : number of non-blocking messages between waits 
    int nb_stride = 5;

    int tag = 103044;
    int ctr;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2*nb_stride*sizeof(MPI_Request));

    memcpy(
        recvbuf + (rdispls[rank] * recv_size),
        sendbuf + (sdispls[rank] * send_size), 
        sendcounts[rank] * send_size);        

    // For each step i
    // exchange among procs stride (i+1) apart
    ctr = 0;
    for (int i = 1; i <= nb_stride && i < num_procs; i++)
    {
        send_proc = rank + i;
        if (send_proc >= num_procs)
            send_proc -= num_procs;
        recv_proc = rank - i;
        if (recv_proc < 0)
            recv_proc += num_procs;

        send_pos = sdispls[send_proc] * send_size;
        recv_pos = rdispls[recv_proc] * recv_size;

        MPI_Isend(sendbuf + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                comm, &(requests[ctr++]));
        MPI_Irecv(recvbuf + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                comm, &(requests[ctr++]));

    }

    if (nb_stride >= num_procs)
    {
        MPI_Waitall(2*(num_procs-1), requests, MPI_STATUSES_IGNORE);
        free(requests);
        return 0;
    }

    int send_idx = nb_stride;
    int recv_idx = nb_stride;
    int idx;
    while (1)
    {
        MPI_Waitany(2*nb_stride, requests, &idx, MPI_STATUSES_IGNORE);

        if (idx == MPI_UNDEFINED)
        {
            break;
        }

        if (idx % 2 == 0 && send_idx < num_procs)
        {
            send_proc = rank + send_idx;
            if (send_proc >= num_procs)
                send_proc -= num_procs;
            send_pos = sdispls[send_proc] * send_size;
            MPI_Isend(sendbuf + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                    comm, &(requests[idx]));
            send_idx++;
        }
        else if (idx % 2 == 1 && recv_idx < num_procs)
        {
            recv_proc = rank - recv_idx;
            if (recv_proc < 0)
                recv_proc += num_procs;
            recv_pos = rdispls[recv_proc] * recv_size;

            MPI_Irecv(recvbuf + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                    comm, &(requests[idx]));
            recv_idx++;
        }
    }

    free(requests);

    return 0;
}

// 2-Step Aggregation (large messages)
// Gather all data to be communicated between nodes
// Send to node+i, recv from node-i
// TODO (For Evelyn to look at sometime?) : 
//     What is the best way to aggregate very large messages?
//     Should we load balance to make sure all processes per node
//         send equal amount of data? (ideally, yes)
//     Should we use S. Lockhart's  'ideal' aggregation, setting
//         a tolerance.  Any message with size < tolerance, aggregate
//         this data with other processes locally.
//     How should we aggregate data when using GPU memory??
int alltoallv_pairwise_loc(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* mpi_comm)
{
    int rank, num_procs;
    int local_rank, PPN; 
    int num_nodes, rank_node;
    MPI_Comm_rank(mpi_comm->global_comm, &rank);
    MPI_Comm_size(mpi_comm->global_comm, &num_procs);
    MPI_Comm_rank(mpi_comm->local_comm, &local_rank);
    MPI_Comm_size(mpi_comm->local_comm, &PPN);
    num_nodes = mpi_comm->num_nodes;
    rank_node = mpi_comm->rank_node;

    const char* send_buffer = (char*) sendbuf;
    char* recv_buffer = (char*) recvbuf;
    int sbytes, rbytes;
    MPI_Type_size(sendtype, &sbytes);
    MPI_Type_size(recvtype, &rbytes);

    int tag = 102913;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    int send_node, recv_node;
    MPI_Status status;

    int final_recvcount = 0;
    for (int i = 0; i < num_procs; i++)
        final_recvcount += recvcounts[i];

    /************************************************
     * Step 1 : Send aggregated data to node
     ***********************************************/
    int sendcount, recvcount;
    int* global_recvcounts = (int*)malloc(num_procs*sizeof(int));
    int global_recvcount = 0;
    // Send to node + i
    // Recv from node - i
    for (int i = 0; i < num_nodes; i++)
    {
        send_node = rank_node + i;
        if (send_node >= num_nodes)
            send_node -= num_nodes;
        recv_node = rank_node - i;
        if (recv_node < 0)
            recv_node += num_nodes;

        MPI_Sendrecv(&(sendcounts[send_node*PPN]), PPN, MPI_INT,
                send_node*PPN+local_rank, tag,
                &(global_recvcounts[recv_node*PPN]), PPN, MPI_INT,
                recv_node*PPN+local_rank, tag,
                mpi_comm->global_comm, &status); 
    }

    int maxrecvcount = final_recvcount;
    if (global_recvcount > maxrecvcount)
        maxrecvcount = global_recvcount;
    char* tmpbuf = (char*)malloc(maxrecvcount*rbytes);
    char* contigbuf = (char*)malloc(maxrecvcount*rbytes);

    // Send to node + i
    // Recv from node - i
    for (int i = 0; i < num_nodes; i++)
    {
        send_node = rank_node + i;
        if (send_node >= num_nodes)
            send_node -= num_nodes;
        recv_node = rank_node - i;
        if (recv_node < 0)
            recv_node += num_nodes;
        send_pos = sdispls[send_node * PPN];
        recv_pos = rdispls[recv_node * PPN];

        sendcount = 0;
        recvcount = 0;
        for (int j = 0; j < PPN; j++)
        {
            sendcount += sendcounts[send_node*PPN+j];
            recvcount += global_recvcounts[recv_node*PPN+j];
        }

        MPI_Sendrecv(sendbuf + send_pos*sbytes, sendcount, 
                sendtype, send_node*PPN + local_rank, tag,
                tmpbuf + recv_pos*rbytes, recvcount, 
                recvtype, recv_node*PPN + local_rank, tag, 
                mpi_comm->global_comm, &status);
    }

    /************************************************
     * Step 2 : Redistribute received data within node
     ************************************************/
    int* ppn_ctr = (int*)malloc(PPN*sizeof(int));
    int* ppn_displs = (int*)malloc((PPN+1)*sizeof(int));
    for (int i = 0; i < PPN; i++)
        ppn_ctr[i] = 0;
    for (int i = 0; i < num_nodes; i++)
        for (int j = 0; j < PPN; j++)
            ppn_ctr[j] += global_recvcounts[i*PPN+j];
    ppn_displs[0] = 0;
    for (int i = 0; i < PPN; i++)
    {
        ppn_displs[i+1] = ppn_displs[i] + ppn_ctr[i];
        ppn_ctr[i] = 0;
    }

    // TODO (for Evelyn to look into?) : 
    //     Currently, re-pack data here
    //     We recv'd data from each node
    //     Now we re-pack it so that it is
    //     ordered by destination process rather
    //     than source node.
    //     Packing can be expensive! Should we
    //     use MPI Datatypes?  Or send num_nodes 
    //     different messages to each of the PPN
    //     local processes?

    int ctr = 0;
    recvcount = 0;
    for (int i = 0; i < num_nodes; i++)
        for (int j = 0; j < PPN; j++)
        {
            recvcount = global_recvcounts[i*PPN+j];
            memcpy(recvbuf + (ppn_displs[j] + ppn_ctr[j])*rbytes,
                    tmpbuf + ctr*rbytes,
                    recvcount*rbytes);
            ctr += recvcount;
            ppn_ctr[j] += recvcount;
        }

    // Send to local_rank + i
    // Recv from local_rank + i
    ctr = 0;
    for (int i = 0; i < PPN; i++)
    {
        send_proc = local_rank + i;
        if (send_proc >= PPN)
            send_proc -= PPN;
        recv_proc = local_rank - i;
        if (recv_proc < 0)
            recv_proc += PPN;

        send_pos = ppn_displs[send_proc] * rbytes;
        recvcount = 0;
        for (int j = 0; j < num_nodes; j++)
            recvcount += recvcounts[j*PPN+i];

        MPI_Sendrecv(recvbuf + send_pos, ppn_ctr[send_proc], recvtype,
                send_proc, tag,
                tmpbuf + ctr*rbytes, recvcount, recvtype,
                recv_proc, tag,
                mpi_comm->local_comm, &status);

        ppn_ctr[recv_proc] = ctr;

        ctr += recvcount;
    }

    for (int i = 0; i < PPN; i++)
    {
        for (int j = 0; j < num_nodes; j++)
        {
            memcpy(recvbuf + rdispls[j*PPN+i]*rbytes,
                    tmpbuf + ppn_ctr[i]*rbytes,
                    recvcounts[j*PPN+i]*rbytes);
            ppn_ctr[i] += recvcounts[j*PPN+i];
        }
    }

    free(ppn_ctr);
    free(ppn_displs);
    free(global_recvcounts);
    free(contigbuf);
    free(tmpbuf);

    return 0;
}

