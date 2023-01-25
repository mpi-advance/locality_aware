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
    return alltoallv_pairwise(sendbuf,
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
    return alltoallv_pairwise_loc(sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf,
        recvcounts,
        rdispls,
        recvtype,
        mpi_comm);
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

        send_pos = sdispls[send_proc] * send_size;
        recv_pos = rdispls[recv_proc] * recv_size;

        MPI_Sendrecv(sendbuf + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                recvbuf + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                comm, &status);
    }
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

