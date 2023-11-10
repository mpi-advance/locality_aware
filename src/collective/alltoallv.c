#include "alltoallv.h"
#include <string.h>
#include <math.h>

#ifdef GPU
#include "heterogeneous/gpu_alltoallv.h"
#endif

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


    cudaMemoryType send_type, recv_type;
    get_mem_types(sendbuf, recvbuf, send_type, recv_type);

    if (send_type == cudaMemoryTypeDevice &&
            recv_type == cudaMemoryTypeDevice)
    {  
        return copy_to_cpu_alltoallv_pairwise(sendbuf,
                sendcounts,
                sdispls,
                sendtype,
                recvbuf,
                recvcounts,
                rdispls,
                recvtype,
                mpi_comm);

    }
    else if (send_type == cudaMemoryTypeDevice ||
            recv_type == cudaMemoryTypeDevice)
    {
        return gpu_aware_alltoallv_pairwise(sendbuf,
                sendcounts,
                sdispls,
                sendtype,
                recvbuf,
                recvcounts,
                rdispls,
                recvtype,
                mpi_comm);
    }
#endif
    return alltoallv_pairwise(
        sendbuf,
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
{//printf("Hello, alltoallv_pairwise__***1\n");fflush(stdout);
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
 
    char* send_buffer = (char*)sendbuf;
    char* recv_buffer = (char*)recvbuf;

#ifdef GPU
cudaMemoryType send_type, recv_type;
cudaPointerAttributes send_mem, recv_mem;

cudaPointerGetAttributes(&send_mem, sendbuf);
cudaPointerGetAttributes(&recv_mem, recvbuf);

int send_ierr = cudaGetLastError();
int recv_ierr = cudaGetLastError();

if (send_ierr == cudaErrorInvalidValue)
    send_type = cudaMemoryTypeHost;
else
    send_type = send_mem.type;

if (recv_ierr == cudaErrorInvalidValue)
    recv_type = cudaMemoryTypeHost;
else
    recv_type = recv_mem.type;
 
    if (send_type == cudaMemoryTypeDevice &&
            recv_type == cudaMemoryTypeDevice){

// GPU to GPU memory transfer
    cudaMemcpy(recv_buffer + rdispls[rank] * recv_size,
               send_buffer + sdispls[rank] * send_size,
               sendcounts[rank] * send_size,
               cudaMemcpyDeviceToDevice);


} else if (send_type == cudaMemoryTypeDevice) {
    // GPU to CPU memory transfer
    cudaMemcpy(recv_buffer + rdispls[rank] * recv_size,
               send_buffer + sdispls[rank] * send_size,
               sendcounts[rank] * send_size,
               cudaMemcpyDeviceToHost);
} else if (recv_type == cudaMemoryTypeDevice) {
    // CPU to GPU memory transfer
    cudaMemcpy(recv_buffer + rdispls[rank] * recv_size,
               send_buffer + sdispls[rank] * send_size,
               sendcounts[rank] * send_size,
               cudaMemcpyHostToDevice);    
    }else{    
#endif

memcpy(
        recv_buffer + (rdispls[rank] * recv_size),
        send_buffer + (sdispls[rank] * send_size), 
        sendcounts[rank] * send_size); 
#ifdef GPU
}
#endif
 
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

    // Send to rank + i
        MPI_Sendrecv(send_buffer + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                recv_buffer + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
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

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);


    MPI_Request* requests = (MPI_Request*)malloc(2*(num_procs-1)*sizeof(MPI_Request));


    char* send_buffer = (char*)sendbuf;
    char* recv_buffer = (char*)recvbuf;

    // For each step i
    // exchange among procs stride (i+1) apart
#ifdef GPU
cudaMemoryType send_type, recv_type;
cudaPointerAttributes send_mem, recv_mem;

cudaPointerGetAttributes(&send_mem, sendbuf);
cudaPointerGetAttributes(&recv_mem, recvbuf);

int send_ierr = cudaGetLastError();
int recv_ierr = cudaGetLastError();

if (send_ierr == cudaErrorInvalidValue)
    send_type = cudaMemoryTypeHost;
else
    send_type = send_mem.type;

if (recv_ierr == cudaErrorInvalidValue)
    recv_type = cudaMemoryTypeHost;
else
    recv_type = recv_mem.type;

if (send_type == cudaMemoryTypeDevice && recv_type == cudaMemoryTypeDevice) {
    // GPU to GPU memory transfer
    cudaMemcpy(recv_buffer + rdispls[rank] * recv_size,
               send_buffer + sdispls[rank] * send_size,
               sendcounts[rank] * send_size,
               cudaMemcpyDeviceToDevice);
} else if (send_type == cudaMemoryTypeDevice) {
    // GPU to CPU memory transfer
    cudaMemcpy(recv_buffer + rdispls[rank] * recv_size,
               send_buffer + sdispls[rank] * send_size,
               sendcounts[rank] * send_size,
               cudaMemcpyDeviceToHost);
} else if (recv_type == cudaMemoryTypeDevice) {
    // CPU to GPU memory transfer
    cudaMemcpy(recv_buffer + rdispls[rank] * recv_size,
               send_buffer + sdispls[rank] * send_size,
               sendcounts[rank] * send_size,
               cudaMemcpyHostToDevice);
} else {
#endif
// CPU to CPU memory transfer
memcpy(recv_buffer + rdispls[rank] * recv_size,
       send_buffer + sdispls[rank] * send_size,
       sendcounts[rank] * send_size);
#ifdef GPU
}
#endif    
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
        MPI_Isend(send_buffer + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                comm, &(requests[i-1]));
        MPI_Irecv(recv_buffer + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                comm, &(requests[num_procs+i-2]));
    }

    MPI_Waitall(2*(num_procs-1), requests, MPI_STATUSES_IGNORE);
    free(requests);

    return 0;
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
/*    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    // Tuning Parameter : number of non-blocking messages between waits 
    int nb_stride = 5;

    int tag = 103044;
    int ctr;
    int send_proc, recv_proc;
    int send_pos, recv_pos;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2*nb_stride*sizeof(MPI_Request));


    char* send_buffer = (char*)sendbuf;
    char* recv_buffer = (char*)recvbuf;

#ifdef GPU
cudaMemoryType send_type, recv_type;
cudaPointerAttributes send_mem, recv_mem;

cudaPointerGetAttributes(&send_mem, sendbuf);
cudaPointerGetAttributes(&recv_mem, recvbuf);

int send_ierr = cudaGetLastError();
int recv_ierr = cudaGetLastError();

if (send_ierr == cudaErrorInvalidValue)
    send_type = cudaMemoryTypeHost;
else
    send_type = send_mem.type;

if (recv_ierr == cudaErrorInvalidValue)
    recv_type = cudaMemoryTypeHost;
else
    recv_type = recv_mem.type;

if (send_type == cudaMemoryTypeDevice && recv_type == cudaMemoryTypeDevice) {

    // GPU to GPU memory transfer
     
    cudaMemcpy(recv_buffer + rdispls[rank] * recv_size,
               send_buffer + sdispls[rank] * send_size,
               sendcounts[rank] * send_size,
               cudaMemcpyDeviceToDevice);
} else if (send_type == cudaMemoryTypeDevice) {
    // GPU to CPU memory transfer
    cudaMemcpy(recv_buffer + rdispls[rank] * recv_size,
               send_buffer + sdispls[rank] * send_size,
               sendcounts[rank] * send_size,
               cudaMemcpyDeviceToHost);
} else if (recv_type == cudaMemoryTypeDevice) {
    // CPU to GPU memory transfer
    cudaMemcpy(recv_buffer + rdispls[rank] * recv_size,
               send_buffer + sdispls[rank] * send_size,
               sendcounts[rank] * send_size,
               cudaMemcpyHostToDevice);
} else {
#endif
// CPU to CPU memory transfer
memcpy(recv_buffer + rdispls[rank] * recv_size,
       send_buffer + sdispls[rank] * send_size,
       sendcounts[rank] * send_size);
#ifdef GPU
}
#endif

 */

/*The alg below is faster than the above alg by 2 seconds; This because cudaMemcpyAsync instead of cudaMemcpy */

int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);
 int nb_stride = 5;
    int tag = 103044;
    int send_proc, recv_proc;
    int send_pos, recv_pos;
    MPI_Status status;

    int total_send_size = 0;
    int total_recv_size = 0;
    
    MPI_Type_size(sendtype, &total_send_size);
    MPI_Type_size(recvtype, &total_recv_size);
 
/*********************/

   // int tag = 103044;
    int ctr;
    //int send_proc, recv_proc;
    //int send_pos, recv_pos;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2*nb_stride*sizeof(MPI_Request));

    char* send_buffer = (char*)sendbuf;
    char* recv_buffer = (char*)recvbuf;
/***************************/
    // Calculate total send size
    for (int i = 0; i < num_procs; i++) {
        total_send_size += sendcounts[i] * total_send_size;
    }

    // Calculate total receive size
    for (int i = 0; i < num_procs; i++) {
        total_recv_size += recvcounts[i] * total_recv_size;
    }

    #ifdef GPU
    cudaMemoryType send_type, recv_type;
    cudaPointerAttributes send_mem, recv_mem;

    cudaPointerGetAttributes(&send_mem, sendbuf);
    cudaPointerGetAttributes(&recv_mem, recvbuf);

    int send_ierr = cudaGetLastError();
    int recv_ierr = cudaGetLastError();

    if (send_ierr == cudaErrorInvalidValue)
        send_type = cudaMemoryTypeHost;
    else
        send_type = send_mem.type;

    if (recv_ierr == cudaErrorInvalidValue)
        recv_type = cudaMemoryTypeHost;
    else
        recv_type = recv_mem.type;

    if (send_type == cudaMemoryTypeDevice && recv_type == cudaMemoryTypeDevice) {
        // GPU to GPU memory transfer asynchronously
        cudaMemcpyAsync(recv_buffer + rdispls[rank] * total_recv_size,
                        send_buffer + sdispls[rank] * total_send_size,
                        sendcounts[rank] * total_send_size,
                        cudaMemcpyDeviceToDevice);
    } else if (send_type == cudaMemoryTypeDevice) {
        // GPU to CPU memory transfer asynchronously
        cudaMemcpyAsync(recv_buffer + rdispls[rank] * total_recv_size,
                        send_buffer + sdispls[rank] * total_send_size,
                        sendcounts[rank] * total_send_size,
                        cudaMemcpyDeviceToHost);
    } else if (recv_type == cudaMemoryTypeDevice) {
        // CPU to GPU memory transfer asynchronously
        cudaMemcpyAsync(recv_buffer + rdispls[rank] * total_recv_size,
                        send_buffer + sdispls[rank] * total_send_size,
                        sendcounts[rank] * total_send_size,
                        cudaMemcpyHostToDevice);
    }
    #endif

    #ifndef GPU
    // CPU to CPU memory transfer
    memcpy(recv_buffer + rdispls[rank] * total_recv_size,
           send_buffer + sdispls[rank] * total_send_size,
           sendcounts[rank] * total_send_size);
    #endif



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

        MPI_Isend(send_buffer + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                comm, &(requests[ctr++]));
        MPI_Irecv(recv_buffer + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
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

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests = (MPI_Request*)malloc(2*nb_stride*sizeof(MPI_Request));

    char* send_buffer = (char*)sendbuf;
    char* recv_buffer = (char*)recvbuf;


#ifdef GPU
cudaMemoryType send_type, recv_type;
cudaPointerAttributes send_mem, recv_mem;

cudaPointerGetAttributes(&send_mem, sendbuf);
cudaPointerGetAttributes(&recv_mem, recvbuf);

int send_ierr = cudaGetLastError();
int recv_ierr = cudaGetLastError();

if (send_ierr == cudaErrorInvalidValue)
    send_type = cudaMemoryTypeHost;
else
    send_type = send_mem.type;

if (recv_ierr == cudaErrorInvalidValue)
    recv_type = cudaMemoryTypeHost;
else
    recv_type = recv_mem.type;

if (send_type == cudaMemoryTypeDevice && recv_type == cudaMemoryTypeDevice) {
    // GPU to GPU memory transfer
    cudaMemcpy(recv_buffer + rdispls[rank] * recv_size,
               send_buffer + sdispls[rank] * send_size,
               sendcounts[rank] * send_size,
               cudaMemcpyDeviceToDevice);
} else if (send_type == cudaMemoryTypeDevice) {
    // GPU to CPU memory transfer
    cudaMemcpy(recv_buffer + rdispls[rank] * recv_size,
               send_buffer + sdispls[rank] * send_size,
               sendcounts[rank] * send_size,
               cudaMemcpyDeviceToHost);
} else if (recv_type == cudaMemoryTypeDevice) {
    // CPU to GPU memory transfer
    cudaMemcpy(recv_buffer + rdispls[rank] * recv_size,
               send_buffer + sdispls[rank] * send_size,
               sendcounts[rank] * send_size,
               cudaMemcpyHostToDevice);
} else {
#endif
// CPU to CPU memory transfer
memcpy(recv_buffer + rdispls[rank] * recv_size,
       send_buffer + sdispls[rank] * send_size,
       sendcounts[rank] * send_size);
#ifdef GPU
}
#endif


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

        MPI_Isend(send_buffer + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                comm, &(requests[ctr++]));
        MPI_Irecv(recv_buffer + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
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
            MPI_Isend(send_buffer + send_pos, sendcounts[send_proc], sendtype, send_proc, tag,
                    comm, &(requests[idx]));
            send_idx++;
        }
        else if (idx % 2 == 1 && recv_idx < num_procs)
        {
            recv_proc = rank - recv_idx;
            if (recv_proc < 0)
                recv_proc += num_procs;
            recv_pos = rdispls[recv_proc] * recv_size;

            MPI_Irecv(recv_buffer + recv_pos, recvcounts[recv_proc], recvtype, recv_proc, tag,
                    comm, &(requests[idx]));
            recv_idx++;
        }
    }

    free(requests);

    return 0;
}

