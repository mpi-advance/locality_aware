#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"
#include "neighborhood/MPIL_Topo.h"
#include "neighborhood/neighborhood_init.h"
#include "persistent/MPIL_Request.h"

int neighbor_alltoallv_init_coll_a2a(const void* sendbuf,
                                     const int sendcounts[],
                                     const int sdispls[],
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcounts[],
                                     const int rdispls[],
                                     MPI_Datatype recvtype,
                                     MPIL_Topo* topo,
                                     MPIL_Comm* comm,
                                     MPIL_Info* info,
                                     MPIL_Request** request_ptr)
{
    MPIL_Request* request;
    init_request(&request);

    int tag;
    MPIL_Comm_tag(comm, &tag);

    init_request(&(request->local_S_request));
    init_request(&(request->local_R_request));
    allocate_requests(topo->outdegree*2, request->local_S_request);
    allocate_requests(topo->indegree*2, request->local_R_request);

    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    const char* send_buffer = (const char*)(sendbuf);
    char* recv_buffer       = (char*)(recvbuf);

    int sbytes, rbytes;
    MPI_Type_size(sendtype, &sbytes);
    MPI_Type_size(recvtype, &rbytes);

    int proc, ierr;

    std::vector<int> coll_sendcounts(num_procs, 0);
    std::vector<int> coll_sdispls(num_procs+1);
    std::vector<int> coll_recvcounts(num_procs, 0);
    std::vector<int> coll_rdispls(num_procs+1);

    int ssize = 0;
    for (int i = 0; i < topo->outdegree; i++)
    {
        coll_sendcounts[topo->destinations[i]] = sendcounts[i];
    }
    coll_sdispls[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        coll_sdispls[i+1] = coll_sdispls[i] + coll_sendcounts[i];
    }
    char* coll_sendbuf = (char*)malloc(coll_sdispls[num_procs]*sbytes);
    int* coll_sindices = (int*)malloc(coll_sdispls[num_procs]*sizeof(int));

    // First, will need to repack sendbuf to tmp_sendbuf
    for (int i = 0; i < topo->outdegree; i++)
    {
        proc = topo->destinations[i];
        MPI_Recv_init(&(coll_sendbuf[coll_sdispls[proc]*sbytes]),
                coll_sendcounts[proc],
                sendtype, 
                0,
                0,
                MPI_COMM_SELF, 
                &(request->local_S_request->requests[i]));
        MPI_Send_init(&(send_buffer[sdispls[i]*sbytes]),
                coll_sendcounts[proc],
                sendtype,
                0, 
                0,
                MPI_COMM_SELF, 
                &(request->local_S_request->requests[topo->outdegree+i]));
    }

    int rsize = 0;
    for (int i = 0; i < topo->indegree; i++)
    {
        coll_recvcounts[topo->sources[i]] = recvcounts[i];
    }
    coll_rdispls[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        coll_rdispls[i+1] = coll_rdispls[i] + coll_recvcounts[i];
    }
    char* coll_recvbuf = (char*)malloc(coll_rdispls[num_procs]*rbytes);
    int* coll_rindices = (int*)malloc(coll_rdispls[num_procs]*sizeof(int));

    // Next will call MPI_Alltoallv_init on repacked data
    ierr = MPIL_Alltoallv_init(coll_sendbuf, coll_sendcounts.data(), coll_sdispls.data(), sendtype,
            coll_recvbuf, coll_recvcounts.data(), coll_rdispls.data(), recvtype, 
            comm, info, &(request->local_L_request));

    // Finally will unpack received data
    for (int i = 0; i < topo->indegree; i++)
    {
        proc = topo->sources[i];
        MPI_Recv_init(&(recv_buffer[rdispls[i]*rbytes]),
                coll_recvcounts[proc],
                recvtype, 
                0,
                0,
                MPI_COMM_SELF, 
                &(request->local_R_request->requests[i]));
        MPI_Send_init(&(coll_recvbuf[coll_rdispls[proc]*rbytes]),
                coll_recvcounts[proc],
                recvtype,
                0, 
                0,
                MPI_COMM_SELF, 
                &(request->local_R_request->requests[topo->indegree + i]));
    }


    request->tmp_sendbuf = coll_sendbuf;
    request->tmp_recvbuf = coll_recvbuf;

    request->start_function = neighbor_a2a_start;
    request->wait_function = neighbor_a2a_wait;

    *request_ptr = request;

    return ierr;
}

int neighbor_a2a_start(MPIL_Request* request)
{
    if (request == NULL)
    {
        return MPI_SUCCESS;
    }

#if defined(GPU)
if (request->gpu_sendbuf)
{
    int gpu_error;
#if defined(APU)
    memcpy(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends);
#else
    gpu_error = gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends, 
            gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif

    // First, start local_S

    if (request->local_S_request->n_msgs)
    {
        MPI_Startall(request->local_S_request->n_msgs, 
                request->local_S_request->requests);
        MPI_Waitall(request->local_S_request->n_msgs, 
                request->local_S_request->requests, MPI_STATUSES_IGNORE);
    }

    MPIL_Start(request->local_L_request);

    return MPI_SUCCESS;
}

int neighbor_a2a_wait(MPIL_Request* request, MPI_Status* status)
{
    if (request == NULL)
    {
        return MPI_SUCCESS;
    }

    MPIL_Wait(request->local_L_request, status);

    if (request->local_R_request->n_msgs)
    {
        MPI_Startall(request->local_R_request->n_msgs, 
                request->local_R_request->requests);
        MPI_Waitall(request->local_R_request->n_msgs, 
                request->local_R_request->requests, MPI_STATUSES_IGNORE);
    }

#if defined(GPU)
    int gpu_error;
if (request->gpu_recvbuf)
{
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->recvbuf, request->size_recvs);
#else
    gpu_error = gpuMemcpyAsync(request->gpu_recvbuf, request->recvbuf, request->size_recvs, 
            gpuMemcpyHostToDevice, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif

    return MPI_SUCCESS;
}


#if defined(MPI4)
int neighbor_alltoallv_init_coll_ag(const void* sendbuffer,
                                         const int sendcounts[],
                                         const int sdispls[],
                                         const long global_sindices[],
                                         MPI_Datatype sendtype,
                                         void* recvbuffer,
                                         const int recvcounts[],
                                         const int rdispls[],
                                         const long global_rindices[],
                                         MPI_Datatype recvtype,
                                         MPIL_Topo* topo,
                                         MPIL_Comm* comm,
                                         MPIL_Info* info,
                                         MPIL_Request** request_ptr)
{
    MPIL_Request* request;
    init_request(&request);
    allocate_requests(1, request);

    int sbytes, rbytes;
    MPI_Type_size(sendtype, &sbytes);
    MPI_Type_size(recvtype, &rbytes);

    std::vector<long> unique_sindices;
    int send_size = 0;
    for (int i = 0; i < topo->outdegree; i++)
        send_size += sendcounts[i];

    std::vector<int> send_idx;
    std::map<long, int> sidx_to_pos;
    for (int i = 0; i < send_size; i++)
    {
        long idx = global_sindices[i];
        if (sidx_to_pos.find(idx) == sidx_to_pos.end())
        {
            sidx_to_pos[idx] = i;
            send_idx.push_back(i);
        }
    }

    request->size_sends = send_idx.size();
    request->send_indices = (int*)malloc(request->size_sends*sizeof(int));
    for (int i = 0; i < request->size_sends; i++)
        request->send_indices[i] = send_idx[i];
    request->tmp_sendbuf = malloc(request->size_sends*sbytes);

    int local_size = unique_sindices.size();
    std::vector<int> proc_sizes(num_procs);
    MPI_Allgather(&local_size, 1, MPI_INT, proc_sizes.data(), 1, MPI_INT, comm->global_comm);

    std::vector<int> proc_displs(num_procs+1);
    proc_displs[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        proc_displs[i+1] = proc_displs[i] + proc_sizes[i];
    }
    int total_size = proc_displs[num_procs];

    std::vector<long> gathered_buf(total_size);
    MPI_Allgatherv(unique_sindices.data(), local_size, MPI_LONG,
        gathered_buf.data(), proc_sizes.data(), proc_displs.data(), MPI_LONG,
        comm->global_comm);
    
    int recv_size = 0;
    for (int i = 0; i < topo->indegree; i++)
        recv_size += recvcounts[i];

    std::map<long, int> ridx_to_pos;
    for (int i = 0; i < recv_size; i++)
    {
        long idx = global_sindices[i];
        if (ridx_to_pos.find(idx) == ridx_to_pos.end())
        {
            ridx_to_pos[idx] = i;
        }
    }
    std::vector<int> recv_idx(recv_size);
    for (int i = 0; i < total_size; i++)
    {
        long recv_idx = gathered_buf[i];
        if (ridx_to_pos.find(idx) != ridx_to_pos.end())
        {
            int pos = ridx_to_pos[idx];
            recv_idx[pos] = i;
        }        
    }

    request->size_recvs = recv_size;
    request->recv_indices = (int*)malloc(recv_size*sizeof(int));
    for (int i = 0; i < request->size_recvs; i++)
        request->recv_indices[i] = recv_idx[i];
    request->tmp_recvbuf = malloc(total_size*rbytes);

    request->send_size = sbytes;
    request->recv_size = rbytes;
    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;

    MPI_Allgatherv_init(request->tmp_sendbuf, request->size_sends, sendtype,
            request->tmp_recvbuf, proc_sizes.data(), proc_displs.data(), recvtype,
            comm->global_comm, MPI_INFO_NULL, &(request->requests[0]));
    

    return MPI_SUCCESS;
}
#endif


int neighbor_ag_start(MPIL_Request* request)
{
    if (request == NULL)
    {
        return MPI_SUCCESS;
    }

#if defined(GPU)
if (request->gpu_sendbuf)
{
    int gpu_error;
#if defined(APU)
    memcpy(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends);
#else
    gpu_error = gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends, 
            gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif

    char* send_buffer = (char*)request->sendbuf;
    char* tmp_send_buffer = (char*)request->tmp_sendbuf;
    for (int i = 0; i < request->size_sends; i++)
    {
        MPI_Sendrecv(&(tmp_send_buffer[i*request->send_size]),
                request->send_size,
                MPI_BYTE,
                0,
                0,
                &(send_buffer[request->send_indices[i]*request->send_size]),
                request->send_size, 
                MPI_BYTE, 
                0, 
                0,
                MPI_COMM_SELF,
                MPI_STATUS_IGNORE);
    }

    MPI_Start(&(request->requests[0]));

    return MPI_SUCCESS;
}

int neighbor_ag_wait(MPIL_Request* request, MPI_Status* status)
{
    if (request == NULL)
    {
        return MPI_SUCCESS;
    }

    MPI_Wait(&(request->requests[0]), MPI_STATUS_IGNORE); 

    char* recv_buffer = (char*)request->recvbuf;
    char* tmp_recv_buffer = (char*)request->tmp_recvbuf;
    for (int i = 0; i < request->size_recvs; i++)
    {
        MPI_Sendrecv(&(recv_buffer[i*request->recv_size]),
                request->recv_size,
                MPI_BYTE,
                0,
                0,
                &(tmp_recv_buffer[request->recv_indices[i]*request->recv_size]),
                request->recv_size, 
                MPI_BYTE, 
                0, 
                0,
                MPI_COMM_SELF,
                MPI_STATUS_IGNORE);
    }

#if defined(GPU)
    int gpu_error;
if (request->gpu_recvbuf)
{
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->recvbuf, request->size_recvs);
#else
    gpu_error = gpuMemcpyAsync(request->gpu_recvbuf, request->recvbuf, request->size_recvs, 
            gpuMemcpyHostToDevice, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif

    return MPI_SUCCESS;
}

