#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"
#include "neighborhood/neighborhood_init.h"
#include "collective/alltoallv_init.h"
#include "string.h"

int neighbor_alltoallv_init_rma_helper(const void* sendbuf,
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
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    MPIL_Request* request;
    init_request(&request);
    
    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int tag;
    MPIL_Comm_tag(comm, &tag);

    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;
    request->send_size = send_bytes;
    request->recv_size = recv_bytes;
    request->n_puts = topo->outdegree;
    request->sdispls = (int*)malloc(topo->outdegree*sizeof(int));
    request->put_displs = (int*)malloc(topo->outdegree*sizeof(int));
    request->put_bytes = (int*)malloc(topo->outdegree*sizeof(int));
    request->put_procs = (int*)malloc(topo->outdegree*sizeof(int));

    int bytes = 0;
    for (int i = 0; i < topo->indegree; i++)
        bytes += (recvcounts[i] * recv_bytes);
    MPIL_Request_win_init(request, recvbuf, bytes, 1, comm->global_comm);

    request->win_array = (char*)recvbuf;
    request->win_alloc = 1;

    for (int i = 0; i < topo->outdegree; i++)
    {
        request->sdispls[i] = sdispls[i] * send_bytes;
        request->put_bytes[i] = sendcounts[i] * send_bytes;
        request->put_procs[i] = topo->destinations[i];
    }


    std::vector<MPI_Request> req(topo->outdegree + topo->indegree);
    for (int i = 0; i < topo->outdegree; i++)
    {
        MPI_Irecv(&(request->put_displs[i]), 1, MPI_INT, topo->destinations[i],
                tag, comm->global_comm, &(req[i]));
    } 
    for (int i = 0; i < topo->indegree; i++)
    {
        MPI_Isend(&(rdispls[i]), 1, MPI_INT, topo->sources[i],
                tag, comm->global_comm, &(req[topo->outdegree+i]));
    }
    if (topo->outdegree + topo->indegree)
    {
        MPI_Waitall(topo->outdegree + topo->indegree,
                req.data(), MPI_STATUSES_IGNORE);
    }
    for (int i = 0; i < topo->outdegree; i++)
    {
        request->put_displs[i] *= recv_bytes;
    }

    *request_ptr = request;
 
    return MPI_SUCCESS;

}


int neighbor_alltoallv_init_rma(const void* sendbuf,
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
    neighbor_alltoallv_init_rma_helper(sendbuf, sendcounts, sdispls, sendtype,
            recvbuf, recvcounts, rdispls, recvtype, topo, comm, info, request_ptr);

    MPIL_Request* request = *request_ptr;
    request->start_function = neighbor_rma_start;
    request->wait_function = neighbor_rma_wait;

    return MPI_SUCCESS;
}


int neighbor_alltoallv_init_pscw(const void* sendbuf,
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
    neighbor_alltoallv_init_rma_helper(sendbuf, sendcounts, sdispls, sendtype,
            recvbuf, recvcounts, rdispls, recvtype, topo, comm, info, request_ptr);

    MPIL_Request* request = *request_ptr;
    request->start_function = neighbor_pscw_start;
    request->wait_function = neighbor_pscw_wait;

    // Make groups for PSCW
    MPI_Group group_world;
    MPI_Comm_group(comm->global_comm, &group_world);
    MPI_Group_incl(group_world, topo->indegree, topo->sources, &(request->src_group));
    MPI_Group_incl(group_world, topo->outdegree, topo->destinations, &(request->dest_group));
    MPI_Group_free(&group_world);

    return MPI_SUCCESS;
}




int neighbor_rma_start(MPIL_Request* request)
{
#if defined(GPU)
if (request->gpu_sendbuf)
{
#if defined(APU)
    memcpy(request->tmp_gpubuf, request->gpu_sendbuf, 
            request->n_puts * request->send_size);
#else
    int gpu_error;
    gpu_error = gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, 
            request->n_puts * request->send_size, gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif
    
    const char*  send_buffer = (const char* )(request->sendbuf);

    MPI_Win_fence(MPI_MODE_NOPRECEDE, request->win);

    for (int i = 0; i < request->n_puts; ++i) {
        MPI_Put(send_buffer + request->sdispls[i],
                request->put_bytes[i],
                MPI_BYTE,
                request->put_procs[i],
                request->put_displs[i],
                request->put_bytes[i],
                MPI_BYTE,
                request->win);
    }

    return MPI_SUCCESS;
}


int neighbor_rma_wait(MPIL_Request* request, MPI_Status* status)
{
    MPI_Win_fence(MPI_MODE_NOSUCCEED, request->win);

#if defined(GPU)
if (request->gpu_recvbuf)
{
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->win_array, request->win_bytes);
#else
    int gpu_error;
    gpu_error = gpuMemcpyAsync(request->gpu_recvbuf, request->win_array, 
            request->win_bytes, gpuMemcpyHostToDevice, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif


    return MPI_SUCCESS;
}
   




int neighbor_pscw_start(MPIL_Request* request)
{
#if defined(GPU)
if (request->gpu_sendbuf)
{
#if defined(APU)
    memcpy(request->tmp_gpubuf, request->gpu_sendbuf, 
            request->n_puts * request->send_size);
#else
    int gpu_error;
    gpu_error = gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, 
            request->n_puts * request->send_size, gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif
    
    const char*  send_buffer = (const char* )(request->sendbuf);

    MPI_Win_post(request->src_group, 0, request->win);

    MPI_Win_start(request->dest_group, 0, request->win);
    for (int i = 0; i < request->n_puts; ++i) {
        MPI_Put(send_buffer + request->sdispls[i],
                request->put_bytes[i],
                MPI_BYTE,
                request->put_procs[i],
                request->put_displs[i],
                request->put_bytes[i],
                MPI_BYTE,
                request->win);
    }
    MPI_Win_complete(request->win);


    return MPI_SUCCESS;
}


int neighbor_pscw_wait(MPIL_Request* request, MPI_Status* status)
{
    MPI_Win_wait(request->win);
#if defined(GPU)
if (request->gpu_recvbuf)
{
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->win_array, request->win_bytes);
#else
    int gpu_error;
    gpu_error = gpuMemcpyAsync(request->gpu_recvbuf, request->win_array, 
            request->win_bytes, gpuMemcpyHostToDevice, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif


    return MPI_SUCCESS;
}
    
