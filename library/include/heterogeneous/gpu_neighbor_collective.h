#ifndef MPI_ADVANCE_GPU_NEIGHBOR_COLLECTIVE_H
#define MPI_ADVANCE_GPU_NEIGHBOR_COLLECTIVE_H

#include "gpu_utils.h"
#include "persistent/MPIL_Request.h"
#include "locality_aware.h"

/************************************************
 ***** GPU-Aware NeighborCollective Wrapper *****
 ***********************************************/
#if defined(GPU_AWARE)
template <typename Ftn, typename... Args>
int gpu_aware_neighbor_collective(Ftn f, Args&&... args)
{
    return f(std::forward<Args>(args)...);
}
#endif


/***************************************************
 ***** Copy-to-CPU Neighbor Collective Wrappers ****
 **************************************************/
template <typename Ftn>
int copy_to_cpu_neighbor_alltoallv(Ftn f,
                                const void* sendbuf,
                                const int sendcounts[],
                                const int sdispls[],
                                MPI_Datatype sendtype,
                                void* recvbuf,
                                const int recvcounts[],
                                const int rdispls[],
                                MPI_Datatype recvtype,
                                MPIL_Topo* topo,
                                MPIL_Comm* comm)
{
    int ierr = 0;
    int gpu_error;

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    // Need to use displs instead of counts because buffer
    // may not be contiguous
    int send_size = 0;
    for (int i = 0; i < topo->outdegree; i++)
    {
        if (sdispls[i] + sendcounts[i] > send_size)
            send_size = sdispls[i] + sendcounts[i];
    }
    int recv_size = 0;
    for (int i = 0; i < topo->indegree; i++)
    {
        if (rdispls[i] + recvcounts[i] > recv_size)
            recv_size = rdispls[i] + recvcounts[i];
    }

    void *cpu_sendbuf, *cpu_recvbuf;
    cpu_sendbuf = malloc(send_size * send_bytes);
    cpu_recvbuf = malloc(recv_size * recv_bytes);

#if defined(APU)
    memcpy(cpu_sendbuf, sendbuf, send_size * send_bytes);
#else
    gpu_error = gpuMemcpyAsync(cpu_sendbuf, sendbuf, 
            send_size * send_bytes,
            gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif

    ierr = f(cpu_sendbuf, sendcounts, sdispls, sendtype,
            cpu_recvbuf, recvcounts, rdispls, recvtype,
            topo, comm);

#if defined(APU)
    memcpy(recvbuf, cpu_recvbuf, recv_size * recv_bytes);
#else
    gpu_error = gpuMemcpyAsync(recvbuf, cpu_recvbuf, 
            recv_size * recv_bytes,
            gpuMemcpyHostToDevice, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif

    free(cpu_sendbuf);
    free(cpu_recvbuf);

    return ierr;
}


template <typename Ftn>
int copy_to_cpu_neighbor_alltoallv_init(Ftn f,
                                const void* sendbuf,
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
                                MPIL_Request** req_ptr)
{
    int ierr = 0;
    int gpu_error;

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    // Need to use displs instead of counts because buffer
    // may not be contiguous
    int send_size = 0;
    for (int i = 0; i < topo->outdegree; i++)
    {
        if (sdispls[i] + sendcounts[i] > send_size)
            send_size = sdispls[i] + sendcounts[i];
    }
    int recv_size = 0;
    for (int i = 0; i < topo->indegree; i++)
    {
        if (rdispls[i] + recvcounts[i] > recv_size)
            recv_size = rdispls[i] + recvcounts[i];
    }

    void *cpu_sendbuf, *cpu_recvbuf;
    MPIL_Alloc(&cpu_sendbuf, send_size * send_bytes);
    MPIL_Alloc(&cpu_recvbuf, recv_size * recv_bytes);

    ierr = f(cpu_sendbuf, sendcounts, sdispls, sendtype,
            cpu_recvbuf, recvcounts, rdispls, recvtype,
            topo, comm, info, req_ptr);

    MPIL_Request* request = *req_ptr;
    request->tmp_gpubuf = cpu_sendbuf;
    request->recvbuf = cpu_recvbuf;
    request->gpu_sendbuf = sendbuf;
    request->gpu_recvbuf = recvbuf;
    request->size_sends = send_size * send_bytes;
    request->size_recvs = recv_size * recv_bytes;

    return ierr;
}


template <typename Ftn>
int copy_to_cpu_neighbor_alltoallv_init_ext(Ftn f,
                                const void* sendbuf,
                                const int sendcounts[],
                                const int sdispls[],
                                const long global_sindices[],
                                MPI_Datatype sendtype,
                                void* recvbuf,
                                const int recvcounts[],
                                const int rdispls[],
                                const long global_rindices[],
                                MPI_Datatype recvtype,
                                MPIL_Topo* topo,
                                MPIL_Comm* comm,
                                MPIL_Info* info,
                                MPIL_Request** req_ptr)
{
    int ierr = 0;
    int gpu_error;

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    // Need to use displs instead of counts because buffer
    // may not be contiguous
    int send_size = 0;
    for (int i = 0; i < topo->outdegree; i++)
    {
        if (sdispls[i] + sendcounts[i] > send_size)
            send_size = sdispls[i] + sendcounts[i];
    }
    int recv_size = 0;
    for (int i = 0; i < topo->indegree; i++)
    {
        if (rdispls[i] + recvcounts[i] > recv_size)
            recv_size = rdispls[i] + recvcounts[i];
    }

    void *cpu_sendbuf, *cpu_recvbuf;
    MPIL_Alloc(&cpu_sendbuf, send_size * send_bytes);
    MPIL_Alloc(&cpu_recvbuf, recv_size * recv_bytes);

    ierr = f(cpu_sendbuf, sendcounts, sdispls, global_sindices, sendtype,
            cpu_recvbuf, recvcounts, rdispls, global_rindices, recvtype,
            topo, comm, info, req_ptr);

    MPIL_Request* request = *req_ptr;
    request->tmp_gpubuf = cpu_sendbuf;
    request->recvbuf = cpu_recvbuf;
    request->gpu_sendbuf = sendbuf;
    request->gpu_recvbuf = recvbuf;
    request->size_sends = send_size * send_bytes;
    request->size_recvs = recv_size * recv_bytes;

    return ierr;
}



#endif
