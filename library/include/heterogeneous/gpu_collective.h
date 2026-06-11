#ifndef MPI_ADVANCE_GPU_COLLECTIVE_H
#define MPI_ADVANCE_GPU_COLLECTIVE_H

#include "gpu_utils.h"

/************************************************
 ***** GPU-Aware Collective Wrapper *************
 ***********************************************/
#if defined(GPU_AWARE)
template <typename Ftn, typename... Args>
int gpu_aware_collective(Ftn f, Args&&... args)
{
    return f(std::forward<Args>(args)...);
}
#endif




/************************************************
 ***** Copy-to-CPU Collective Wrappers **********
 ***********************************************/
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

    void* cpu_sendbuf = MPIL_Alloc(count*bytes);
    void* cpu_recvbuf = MPIL_Alloc(count*bytes);

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

    void* cpu_sendbuf = MPIL_Alloc(sendcount*send_bytes);
    void* cpu_recvbuf = MPIL_Alloc(recvcount*num_procs*recv_bytes);

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

    void* cpu_sendbuf = MPIL_Alloc(sendcount*num_procs*send_bytes);
    void* cpu_recvbuf = MPIL_Alloc(recvcount*num_procs*recv_bytes);

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

    void* cpu_sendbuf = MPIL_Alloc(sendsize*send_bytes);
    void* cpu_recvbuf = MPIL_Alloc(recvsize*recv_bytes);

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




/************************************************
 ***** Copy-to-CPU Persistent Wrappers **********
 ***********************************************/

template<typename Ftn>
int copy_to_cpu_allreduce_init(Ftn f,
                const void* sendbuf,
                void* recvbuf,
                int count,
                MPI_Datatype datatype,
                MPI_Op op,
                MPIL_Comm* comm,
                MPIL_Info* info,
                MPIL_Request** req_ptr)
{
    int ierr = 0;
    int gpu_error = 0;

    int bytes;
    MPI_Type_size(datatype, &bytes);

    void *cpu_sendbuf, *cpu_recvbuf;
    MPIL_Alloc(&cpu_sendbuf, count*bytes);
    MPIL_Alloc(&cpu_recvbuf, count*bytes);

    ierr = f(cpu_sendbuf, cpu_recvbuf, count,
            datatype, op, comm, info, req_ptr);

    MPIL_Request* request = *req_ptr;
    request->tmp_gpubuf = cpu_sendbuf;
    request->gpu_sendbuf = sendbuf;
    request->gpu_recvbuf = recvbuf;
    request->size_sends = count*bytes;
    request->size_recvs = count*bytes;

    return ierr;
}

template <typename Ftn>
int copy_to_cpu_allgather_init(Ftn f,
                const void* sendbuf,
                const int sendcount,
                MPI_Datatype sendtype,
                void* recvbuf,
                const int recvcount,
                MPI_Datatype recvtype,
                MPIL_Comm* comm,
                MPIL_Info* info,
                MPIL_Request** req_ptr)
{
    int ierr = 0;
    int gpu_error;

    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int total_bytes_s = sendcount * send_bytes;
    int total_bytes_r = recvcount * recv_bytes * num_procs;

    void* cpu_sendbuf, *cpu_recvbuf;
    MPIL_Alloc(&cpu_sendbuf, total_bytes_s);
    MPIL_Alloc(&cpu_recvbuf, total_bytes_r);

    ierr = f(cpu_sendbuf, sendcount, sendtype,
            cpu_recvbuf, recvcount, recvtype, comm, info, req_ptr);


    MPIL_Request* request = *req_ptr;
    request->tmp_gpubuf = cpu_sendbuf;
    request->gpu_sendbuf = sendbuf;
    request->gpu_recvbuf = recvbuf;
    request->size_sends = total_bytes_s;
    request->size_recvs = total_bytes_r;

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

    int total_bytes_s = sendcount * send_bytes * num_procs;
    int total_bytes_r = recvcount * recv_bytes * num_procs;

    void* cpu_sendbuf, *cpu_recvbuf;
    MPIL_Alloc(&cpu_sendbuf, total_bytes_s);
    MPIL_Alloc(&cpu_recvbuf, total_bytes_r);

    ierr = f(cpu_sendbuf, sendcount, sendtype,
            cpu_recvbuf, recvcount, recvtype, comm, info, req_ptr);

    MPIL_Request* request = *req_ptr;
    request->tmp_gpubuf = cpu_sendbuf;
    request->gpu_sendbuf = sendbuf;
    request->gpu_recvbuf = recvbuf;
    request->size_sends = total_bytes_s;
    request->size_recvs = total_bytes_r;

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

    int total_bytes_s = sendsize*send_bytes;
    int total_bytes_r = recvsize*recv_bytes;

    void* cpu_sendbuf, *cpu_recvbuf;
    MPIL_Alloc(&cpu_sendbuf, total_bytes_s);
    MPIL_Alloc(&cpu_recvbuf, total_bytes_r);


    ierr = f(cpu_sendbuf, sendcounts, sdispls, sendtype,
            cpu_recvbuf, recvcounts, rdispls, recvtype, comm,
            info, req_ptr);

    MPIL_Request* request = *req_ptr;
    request->tmp_gpubuf = cpu_sendbuf;
    request->gpu_sendbuf = sendbuf;
    request->gpu_recvbuf = recvbuf;
    request->size_sends = total_bytes_s;
    request->size_recvs = total_bytes_r;

    return ierr;
}





#endif
