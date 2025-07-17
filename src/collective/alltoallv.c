#include "alltoallv.h"
#include <string.h>
#include <math.h>

#ifdef GPU
#include "heterogeneous/gpu_alltoallv.h"
#endif

// Default alltoallv is pairwise
AlltoallvMethod mpix_alltoallv_implementation = ALLTOALLV_PAIRWISE;

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
#ifdef GPU
#ifdef GPU_AWARE
    return gpu_aware_alltoallv_pairwise(sendbuf,
            sendcounts,
            sdispls,
            sendtype,
            recvbuf,
            recvcounts,
            rdispls,
            recvtype,
            mpi_comm);
#endif
#endif
    alltoallv_ftn method;

    switch (mpix_alltoallv_implementation)
    {
        case ALLTOALLV_PAIRWISE:
            method = alltoallv_pairwise;
            break;
        case ALLTOALLV_NONBLOCKING:
            method = alltoallv_nonblocking;
            break;
        case ALLTOALLV_BATCH:
            method = alltoallv_batch;
            break;
        case ALLTOALLV_BATCH_ASYNC:
            method = alltoallv_batch_async;
            break;
        case ALLTOALLV_PMPI:
            method = alltoallv_pmpi;
            break;
        default:
            method = alltoallv_pmpi;
            break;
    }

    return method(
        sendbuf,
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
        MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag;
    MPIX_Comm_tag(comm, &tag);

    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);
 
    char* send_buffer = (char*)sendbuf;
    char* recv_buffer = (char*)recvbuf;

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

        MPI_Sendrecv(send_buffer + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                recv_buffer + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                comm->global_comm, &status);
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
        MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    if (num_procs <= 1)
        alltoallv_pairwise(sendbuf, sendcounts, sdispls, sendtype,
                recvbuf, recvcounts, rdispls, recvtype, comm);

    int tag;
    MPIX_Comm_tag(comm, &tag);

    int send_proc, recv_proc;
    int send_pos, recv_pos;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2*num_procs*sizeof(MPI_Request));

    char* send_buffer = (char*)sendbuf;
    char* recv_buffer = (char*)recvbuf;

    // For each step i
    // exchange among procs stride (i+1) apart
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

        MPI_Isend(send_buffer + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                comm->global_comm, &(requests[i]));
        MPI_Irecv(recv_buffer + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                comm->global_comm, &(requests[num_procs+i]));
    }

    MPI_Waitall(2*num_procs, requests, MPI_STATUSES_IGNORE);

    free(requests);

    return 0;
}

int alltoallv_batch(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    // Tuning Parameter : number of non-blocking messages between waits 
    int nb_stride = 5;
    if (nb_stride >= num_procs)
        alltoallv_nonblocking(sendbuf, sendcounts, sdispls, sendtype,
                recvbuf, recvcounts, rdispls, recvtype, comm);

    int tag;
    MPIX_Comm_tag(comm, &tag);

    int ctr;
    int send_proc, recv_proc;
    int send_pos, recv_pos;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2*nb_stride*sizeof(MPI_Request));

    char* send_buffer = (char*)sendbuf;
    char* recv_buffer = (char*)recvbuf;

    // For each step i
    // exchange among procs stride (i+1) apart
    ctr = 0;
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

        MPI_Isend(send_buffer + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                comm->global_comm, &(requests[ctr++]));
        MPI_Irecv(recv_buffer + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                comm->global_comm, &(requests[ctr++]));

        if ((i+1) % nb_stride == 0)
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

int alltoallv_batch_async(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    // Tuning Parameter : number of non-blocking messages between waits 
    int nb_stride = 5;
    if (nb_stride >= num_procs)
        return alltoallv_nonblocking(sendbuf, sendcounts, sdispls, sendtype, 
                recvbuf, recvcounts, rdispls, recvtype, comm);

    int tag;
    MPIX_Comm_tag(comm, &tag);

    int send_proc, recv_proc;
    int send_pos, recv_pos;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2*nb_stride*sizeof(MPI_Request));

    char* send_buffer = (char*)sendbuf;
    char* recv_buffer = (char*)recvbuf;

    // For each step i
    // exchange among procs stride (i+1) apart
    int send_idx = 0;
    int recv_idx = 0;
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

        MPI_Isend(send_buffer + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                comm->global_comm, &(requests[send_idx++]));
        MPI_Irecv(recv_buffer + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                comm->global_comm, &(requests[nb_stride + recv_idx++]));

        if ((i+1) >= nb_stride)
        {
            MPI_Waitany(nb_stride, requests, &send_idx, MPI_STATUSES_IGNORE);
            MPI_Waitany(nb_stride, &(requests[nb_stride]), &recv_idx, MPI_STATUSES_IGNORE);
        }
    }

    MPI_Waitall(2*nb_stride, requests, MPI_STATUSES_IGNORE);

    free(requests);

    return 0;
}


// Calls underlying MPI implementation
int alltoallv_pmpi(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype,
            recvbuf, recvcounts, rdispls, recvtype, comm->global_comm);
}
