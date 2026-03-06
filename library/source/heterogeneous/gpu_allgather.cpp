#include "heterogeneous/gpu_allgather.h"
#include "heterogeneous/gpu_utils.h"
#include "collective/allgather.h"
#include "locality_aware.h"

// ASSUMES 1 CPU CORE PER GPU (Standard for applications)
int gpu_aware_allgather(allgather_helper_ftn f,
                        const void* sendbuf,
                        int sendcount,
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        int recvcount,
                        MPI_Datatype recvtype,
                        MPIL_Comm* comm)
{
    return f(sendbuf, sendcount, sendtype, 
                recvbuf, recvcount, recvtype, comm,
                MPIL_GPU_Alloc, MPIL_GPU_Free);
}

int gpu_aware_allgather_ring(const void* sendbuf,
                        int sendcount,
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        int recvcount,
                        MPI_Datatype recvtype,
                        MPIL_Comm* comm)
{
    return gpu_aware_allgather(allgather_ring_helper,
                            sendbuf, 
                            sendcount,
                            sendtype,
                            recvbuf,
                            recvcount,
                            recvtype,
                            comm);
}

int gpu_aware_allgather_bruck(const void* sendbuf,
                        int sendcount,
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        int recvcount,
                        MPI_Datatype recvtype,
                        MPIL_Comm* comm)
{
    return gpu_aware_allgather(allgather_bruck_helper,
                            sendbuf,
                            sendcount,
                            sendtype,
                            recvbuf,
                            recvcount,
                            recvtype,
                            comm);
}

int gpu_aware_allgather_pmpi(const void* sendbuf,
                        int sendcount,
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        int recvcount,
                        MPI_Datatype recvtype,
                        MPIL_Comm* comm)
{
    return PMPI_Allgather(sendbuf, sendcount, sendtype,
                recvbuf, recvcount, recvtype, comm->global_comm);
}

int copy_to_cpu_allgather(allgather_helper_ftn f,
                        const void* sendbuf,
                        int sendcount,
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        int recvcount,
                        MPI_Datatype recvtype,
                        MPIL_Comm* comm)
{
    int ierr = 0;

    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);
    
    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    int send_bytes = sendcount*send_size;
    int recv_bytes = recvcount*recv_size*num_procs;

    // gpuMalloc is too expensive for single allgather
    void* cpu_sendbuf = malloc(send_bytes);
    void* cpu_recvbuf = malloc(recv_bytes);

#if defined(APU)
    memcpy(cpu_sendbuf, sendbuf, send_bytes);
#else
    gpuMemcpy(cpu_sendbuf, sendbuf, send_bytes, gpuMemcpyDeviceToHost);
    gpuStreamSynchronize(0);
#endif 

    ierr += f(cpu_sendbuf, sendcount, sendtype, cpu_recvbuf,
            recvcount, recvtype, comm, MPIL_Alloc, MPIL_Free);

#if defined(APU)
    memcpy(recvbuf, cpu_recvbuf, recv_bytes);
#else
    gpuMemcpy(recvbuf, cpu_recvbuf, recv_bytes, gpuMemcpyHostToDevice);
    gpuStreamSynchronize(0);
#endif

    free(cpu_sendbuf);
    free(cpu_recvbuf);

    gpuDeviceSynchronize();

    return ierr;
}

int copy_to_cpu_allgather_ring(const void* sendbuf,
                        int sendcount,
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        int recvcount,
                        MPI_Datatype recvtype,
                        MPIL_Comm* comm)
{
    return copy_to_cpu_allgather(allgather_ring_helper,
                                sendbuf,
                                sendcount,
                                sendtype,
                                recvbuf,
                                recvcount,
                                recvtype,
                                comm);
}

int copy_to_cpu_allgather_bruck(const void* sendbuf,
                        int sendcount,
                        MPI_Datatype sendtype,
                        void* recvbuf,
                        int recvcount,
                        MPI_Datatype recvtype,
                        MPIL_Comm* comm)
{
    return copy_to_cpu_allgather(allgather_bruck_helper,
                                sendbuf,
                                sendcount,
                                sendtype,
                                recvbuf,
                                recvcount,
                                recvtype,
                                comm);
}


int copy_to_cpu_allgather_pmpi(const void* sendbuf,
                                int sendcount,
                                MPI_Datatype sendtype,
                                void* recvbuf, 
                                int recvcount,
                                MPI_Datatype recvtype,
                                MPIL_Comm* comm)
{
    int ierr = 0;

    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    int send_bytes = sendcount*send_size;
    int recv_bytes = recvcount*recv_size*num_procs;

    // gpuMalloc is too expensive for single allgather
    void* cpu_sendbuf = malloc(send_bytes);
    void* cpu_recvbuf = malloc(recv_bytes);

#if defined(APU)
    memcpy(cpu_sendbuf, sendbuf, send_bytes);
#else
    gpuMemcpy(cpu_sendbuf, sendbuf, send_bytes, gpuMemcpyDeviceToHost);
    gpuStreamSynchronize(0); // needed on tuolumne
#endif

    ierr += PMPI_Allgather(cpu_sendbuf, sendcount, sendtype,
                cpu_recvbuf, recvcount, recvtype, comm->global_comm);

#if defined(APU)
    memcpy(recvbuf, cpu_recvbuf, recv_bytes);
#else
    gpuMemcpy(recvbuf, cpu_recvbuf, recv_bytes, gpuMemcpyHostToDevice);
    gpuStreamSynchronize(0); // needed on tuolumne
#endif

    free(cpu_sendbuf);
    free(cpu_recvbuf);

    return ierr;   
}
