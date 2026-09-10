#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"
#include "neighborhood/MPIL_Topo.h"
#include "neighborhood/neighborhood_init.h"
#include "persistent/MPIL_Request.h"
#include "heterogeneous/gpu_utils.h"
#include <map>

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
    char* coll_sendbuf;
    MPIL_Alloc((void**)&coll_sendbuf, coll_sdispls[num_procs]*sbytes);

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
    char* coll_recvbuf;
    MPIL_Alloc((void**)&coll_recvbuf, coll_rdispls[num_procs]*rbytes);

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
    request->free_ftn = MPIL_Free;

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


