#ifndef MPI_ADVANCE_GPU_COLLECTIVE_H
#define MPI_ADVANCE_GPU_COLLECTIVE_H

#include "gpu_utils.h"

#if defined(GPU_AWARE)
template <typename Ftn, typename... Args>
int gpu_aware_collective(Ftn f, Args&&... args)
{
    return f(std::forward<Args>(args)...);
}
#endif

template<typename Ftn>
int copy_to_cpu_allreduce(Ftn f,
                const void* sendbuf,
                void* recvbuf,
                int count,
                MPI_Datatype datatype,
                MPI_Op op,
                MPIL_Comm* comm)
{
    int ierr = 0;
    int gpu_error = 0;

    int bytes;
    MPI_Type_size(datatype, &bytes);

    void* cpu_sendbuf = malloc(count*bytes);
    void* cpu_recvbuf = malloc(count*bytes);

#if defined(APU)
    memcpy(cpu_sendbuf, sendbuf, count*bytes);
#else
    gpu_error = gpuMemcpyAsync(cpu_sendbuf, sendbuf, count*bytes,
            gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif

    ierr = f(cpu_sendbuf, cpu_recvbuf, count,
            datatype, op, comm);

#if defined(APU)
    memcpy(recvbuf, cpu_recvbuf, count*bytes);
#else
    gpu_error = gpuMemcpyAsync(recvbuf, cpu_recvbuf, count*bytes,
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
int copy_to_cpu_allgather(Ftn f,
                const void* sendbuf,
                const int sendcount,
                MPI_Datatype sendtype,
                void* recvbuf,
                const int recvcount,
                MPI_Datatype recvtype,
                MPIL_Comm* comm)
{
    int ierr = 0;
    int gpu_error;

    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    void* cpu_sendbuf = malloc(sendcount*send_bytes);
    void* cpu_recvbuf = malloc(recvcount*num_procs*recv_bytes);

#if defined(APU)
    memcpy(cpu_sendbuf, sendbuf, sendcount*send_bytes);
#else
    gpu_error = gpuMemcpyAsync(cpu_sendbuf, sendbuf, sendcount*send_bytes,
            gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif

    ierr = f(cpu_sendbuf, sendcount, sendtype,
            cpu_recvbuf, recvcount, recvtype, comm);

#if defined(APU)
    memcpy(recvbuf, cpu_recvbuf, recvcount*num_procs*recvbytes);
#else
    gpu_error = gpuMemcpyAsync(recvbuf, cpu_recvbuf, 
            recvcount*num_procs*recv_bytes,
            gpuMemcpyHostToDevice, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif

    free(cpu_sendbuf);
    free(cpu_recvbuf);

    return MPI_SUCCESS;
}

template <typename Ftn>
int copy_to_cpu_alltoall(Ftn f,
        const void* sendbuf,
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcount,
        MPI_Datatype recvtype,
        MPIL_Comm* comm)
{
    int ierr = 0;
    int gpu_error;

    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    void* cpu_sendbuf = malloc(sendcount*num_procs*send_bytes);
    void* cpu_recvbuf = malloc(recvcount*num_procs*recv_bytes);

#if defined(APU)
    memcpy(cpu_sendbuf, sendbuf, sendcount*num_procs*send_bytes);
#else
    gpu_error = gpuMemcpyAsync(cpu_sendbuf, sendbuf, 
            sendcount*num_procs*send_bytes,
            gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif

    ierr = f(cpu_sendbuf, sendcount, sendtype,
            cpu_recvbuf, recvcount, recvtype, comm);

#if defined(APU)
    memcpy(recvbuf, cpu_recvbuf, recvcount*num_procs*recvbytes);
#else
    gpu_error = gpuMemcpyAsync(recvbuf, cpu_recvbuf, 
            recvcount*num_procs*recv_bytes,
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
int copy_to_cpu_alltoallv(Ftn f,
            const void* sendbuf,
            const int sendcounts[],
            const int sdispls[],
            MPI_Datatype sendtype,
            void* recvbuf,
            const int recvcounts[],
            const int rdispls[],
            MPI_Datatype recvtype,
            MPIL_Comm* comm)
{
    int ierr = 0;
    int gpu_error;

    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int sendsize = 0;
    int recvsize = 0;
    for (int i = 0; i < num_procs; i++)
    {
        sendsize += sendcounts[i];
        recvsize += recvcounts[i];
    }

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    void* cpu_sendbuf = malloc(sendsize*send_bytes);
    void* cpu_recvbuf = malloc(recvsize*recv_bytes);

#if defined(APU)
    memcpy(cpu_sendbuf, sendbuf, sendsize*send_bytes);
#else
    gpu_error = gpuMemcpyAsync(cpu_sendbuf, sendbuf, 
            sendsize*send_bytes,
            gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif

    ierr = f(cpu_sendbuf, sendcounts, sdispls, sendtype,
            cpu_recvbuf, recvcounts, rdispls, recvtype, comm);

#if defined(APU)
    memcpy(recvbuf, cpu_recvbuf, recvsize*recvbytes);
#else
    gpu_error = gpuMemcpyAsync(recvbuf, cpu_recvbuf, 
            recvsize*recv_bytes,
            gpuMemcpyHostToDevice, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif

    free(cpu_sendbuf);
    free(cpu_recvbuf);

    return ierr;
}

#endif
