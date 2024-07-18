#include "collective/alltoallv.h"
#include "collective/collective.h"
#include "gpu_alltoallv.h"
#include "/g/g92/enamug/install/include/caliper/cali.h"
// ASSUMES 1 CPU CORE PER GPU (Standard for applications)

int gpu_aware_alltoallv(alltoallv_ftn f,
        const void* sendbuf, 
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{//printf("Hello, gpu_aware_alltoallv__***1\n");fflush(stdout);
    return f(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
            comm->global_comm);
//printf("Hello, gpu_aware_alltoallv__***2\n");fflush(stdout);
}


int gpu_aware_alltoallv_waitany(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return gpu_aware_alltoallv(alltoallv_waitany,
        sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf,
        recvcounts,
        rdispls,
        recvtype,
        comm);
}


int gpu_aware_alltoallv_pairwise(const void* sendbuf, 
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return gpu_aware_alltoallv(alltoallv_pairwise,
        sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf,
        recvcounts,
        rdispls,
        recvtype,
        comm);

}

int gpu_aware_alltoallv_nonblocking(const void* sendbuf, 
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return gpu_aware_alltoallv(alltoallv_nonblocking,
        sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf, 
        recvcounts,
        rdispls,
        recvtype,
        comm);


}

int copy_to_cpu_alltoallv(alltoallv_ftn f,
        const void* sendbuf, 
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int ierr = 0;

    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int sendcount = 0;
    int recvcount = 0;
    for (int i = 0; i < num_procs; i++)
    {
        sendcount += sendcounts[i];
        recvcount += recvcounts[i];
    }

    int total_bytes_s = sendcount * send_bytes;
    int total_bytes_r = recvcount * recv_bytes;

    char* cpu_sendbuf;
    char* cpu_recvbuf;
    cudaMallocHost((void**)&cpu_sendbuf, total_bytes_s);
    cudaMallocHost((void**)&cpu_recvbuf, total_bytes_r);

    // Copy from GPU to CPU
    ierr += gpuMemcpy(cpu_sendbuf, sendbuf, total_bytes_s, gpuMemcpyDeviceToHost);

    // Collective Among CPUs
    ierr += f(cpu_sendbuf, sendcounts, sdispls, sendtype, 
            cpu_recvbuf, recvcounts, rdispls, recvtype, comm->global_comm);

    // Copy from CPU to GPU
    ierr += gpuMemcpy(recvbuf, cpu_recvbuf, total_bytes_r, gpuMemcpyHostToDevice);

    cudaFreeHost(cpu_sendbuf);
    cudaFreeHost(cpu_recvbuf);
    
    return ierr;
}


/*No mallocs in this version*/
int copy_to_cpu_alltoallv_extra(alltoallv_ftn f,
        const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm,char* cpu_recvbuf,char* cpu_sendbuf)
/* add these 2 params here; char* cpu_sendbuf;
    char* cpu_recvbuf*/
{
    int ierr = 0;

    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int sendcount = 0;
    int recvcount = 0;
    for (int i = 0; i < num_procs; i++)
    {
        sendcount += sendcounts[i];
        recvcount += recvcounts[i];
    }

    int total_bytes_s = sendcount * send_bytes;
    int total_bytes_r = recvcount * recv_bytes;

    // Copy from GPU to CPU
    CALI_MARK_BEGIN("Copy from GPU to CPU"); 
    ierr += gpuMemcpy(cpu_sendbuf, sendbuf, total_bytes_s, gpuMemcpyDeviceToHost);
    CALI_MARK_END("Copy from GPU to CPU");

    CALI_MARK_BEGIN("Collective Among CPUs ");
    // Collective Among CPUs
    ierr += f(cpu_sendbuf, sendcounts, sdispls, sendtype,
            cpu_recvbuf, recvcounts, rdispls, recvtype, comm->global_comm);

     CALI_MARK_END("Collective Among CPUs ");


    // Copy from CPU to GPU
     CALI_MARK_BEGIN("Copy from CPU to GPU");  
    ierr += gpuMemcpy(recvbuf, cpu_recvbuf, total_bytes_r, gpuMemcpyHostToDevice);
    
     CALI_MARK_END("Copy from CPU to GPU");

return ierr;
}



int copy_to_cpu_alltoallv_pairwise(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return copy_to_cpu_alltoallv(alltoallv_pairwise,
        sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf,
        recvcounts,
        rdispls,
        recvtype,
        comm);
}

int copy_to_cpu_alltoallv_nonblocking(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return copy_to_cpu_alltoallv(alltoallv_nonblocking,
        sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf,
        recvcounts,
        rdispls,
        recvtype,
        comm);
}


int copy_to_cpu_alltoallv_pairwise_extra(const void* sendbuf, 
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm,char* cpu_recvbuf,char* cpu_sendbuf)
{
    return copy_to_cpu_alltoallv_extra(alltoallv_pairwise,
        sendbuf, 
        sendcounts,
        sdispls,
        sendtype,
        recvbuf, 
        recvcounts,
        rdispls,
        recvtype,
        comm, cpu_recvbuf, cpu_sendbuf);

}



int copy_to_cpu_alltoallv_MPI_Alltoallv_extra(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm,char* cpu_recvbuf,char* cpu_sendbuf)
{
    return copy_to_cpu_alltoallv_extra(MPI_Alltoallv ,
        sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf,
        recvcounts,
        rdispls,
        recvtype,
        comm, cpu_recvbuf, cpu_sendbuf);

}


int copy_to_cpu_alltoallv_nonblocking_extra(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm,char* cpu_recvbuf,char* cpu_sendbuf)
{
    return copy_to_cpu_alltoallv_extra(alltoallv_nonblocking,
        sendbuf,
        sendcounts,
        sdispls,
        sendtype,
        recvbuf,
        recvcounts,
        rdispls,
        recvtype,
        comm, cpu_recvbuf, cpu_sendbuf);

}

int threaded_alltoallv_pairwise(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int ierr = 0;

    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int sendcount = 0;
    int recvcount = 0;
    for (int i = 0; i < num_procs; i++)
    {
        sendcount += sendcounts[i];
        recvcount += recvcounts[i];
    }

    int total_bytes_s = sendcount * send_bytes;
    int total_bytes_r = recvcount * recv_bytes;

    char* cpu_sendbuf;
    char* cpu_recvbuf;
    cudaMallocHost((void**)&cpu_sendbuf, total_bytes_s);
    cudaMallocHost((void**)&cpu_recvbuf, total_bytes_r);

    // Copy from GPU to CPU
    ierr += gpuMemcpy(cpu_sendbuf, sendbuf, total_bytes_s, gpuMemcpyDeviceToHost);

    memcpy(cpu_recvbuf + (rdispls[rank] * recv_bytes),
        cpu_sendbuf + (sdispls[rank] * send_bytes),
        sendcounts[rank] * send_bytes);
 
/*
    int* ordered_sends = (int*)malloc(num_procs*sizeof(int));
    int* ordered_recvs = (int*)malloc(num_procs*sizeof(int));
    sort(num_procs, ordered_sends, sendcounts);
    sort(num_procs, ordered_recvs, recvcounts);
*/
#pragma omp parallel shared(cpu_sendbuf, cpu_recvbuf)
{
    MPI_Status status;
    int tag = 102944;
    int send_proc, recv_proc;
    int send_pos, recv_pos;

    int n_msgs = num_procs - 1;
    int thread_id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();

    int n_msgs_per_thread = n_msgs / num_threads;
    int extra_msgs = n_msgs % num_threads;
    int thread_n_msgs = n_msgs_per_thread;
    if (extra_msgs > thread_id)
        thread_n_msgs++;

    if (thread_n_msgs)
    {
        int idx = thread_id + 1;
        for (int i = 0; i < thread_n_msgs; i++)
        {
            send_proc = rank + idx;
            if (send_proc >= num_procs)
                send_proc -= num_procs;
            recv_proc = rank - idx;
            if (recv_proc < 0)
                recv_proc += num_procs;
            send_pos = sdispls[send_proc] * send_bytes;
            recv_pos = rdispls[recv_proc] * recv_bytes;

            MPI_Sendrecv(cpu_sendbuf + send_pos,
                    sendcounts[send_proc],
                    sendtype, 
                    send_proc,
                    tag,
                    cpu_recvbuf + recv_pos,
                    recvcounts[recv_proc],
                    recvtype,
                    recv_proc,
                    tag,
                    comm->global_comm,
                    &status);

            idx += num_threads;
        }
    }
} 

    ierr += gpuMemcpy(recvbuf, cpu_recvbuf, total_bytes_r, gpuMemcpyHostToDevice);

/*
    free(ordered_sends);
    free(ordered_recvs);
*/

    cudaFreeHost(cpu_sendbuf);
    cudaFreeHost(cpu_recvbuf);

    return ierr;
}

int threaded_alltoallv_nonblocking(const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    int ierr = 0;

    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int sendcount = 0;
    int recvcount = 0;
    for (int i = 0; i < num_procs; i++)
    {
        sendcount += sendcounts[i];
        recvcount += recvcounts[i];
    }

    int total_bytes_s = sendcount * send_bytes;
    int total_bytes_r = recvcount * recv_bytes;

    char* cpu_sendbuf;
    char* cpu_recvbuf;
    cudaMallocHost((void**)&cpu_sendbuf, total_bytes_s);
    cudaMallocHost((void**)&cpu_recvbuf, total_bytes_r);

    // Copy from GPU to CPU
    ierr += gpuMemcpy(cpu_sendbuf, sendbuf, total_bytes_s, gpuMemcpyDeviceToHost);

    memcpy(cpu_recvbuf + (rdispls[rank] * recv_bytes),
        cpu_sendbuf + (sdispls[rank] * send_bytes),
        sendcounts[rank] * send_bytes);
 
/*
    int* ordered_sends = (int*)malloc(num_procs*sizeof(int));
    int* ordered_recvs = (int*)malloc(num_procs*sizeof(int));
    sort(num_procs, ordered_sends, sendcounts);
    sort(num_procs, ordered_recvs, recvcounts);
*/
#pragma omp parallel shared(cpu_sendbuf, cpu_recvbuf)
{
    int tag = 102944;
    int send_proc, recv_proc;
    int send_pos, recv_pos;

    int n_msgs = num_procs - 1;
    int thread_id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();

    int n_msgs_per_thread = n_msgs / num_threads;
    int extra_msgs = n_msgs % num_threads;
    int thread_n_msgs = n_msgs_per_thread;
    if (extra_msgs > thread_id)
        thread_n_msgs++;

    if (thread_n_msgs)
    {
        MPI_Request* requests = (MPI_Request*)malloc(2*thread_n_msgs*sizeof(MPI_Request));

        int idx = thread_id + 1;
        for (int i = 0; i < thread_n_msgs; i++)
        {
            send_proc = rank + idx;
            if (send_proc >= num_procs)
                send_proc -= num_procs;
            recv_proc = rank - idx;
            if (recv_proc < 0)
                recv_proc += num_procs;
            send_pos = sdispls[send_proc] * send_bytes;
            recv_pos = rdispls[recv_proc] * recv_bytes;

            MPI_Isend(cpu_sendbuf + send_pos,
                    sendcounts[send_proc],
                    sendtype,
                    send_proc,
                    tag,
                    comm->global_comm,
                    &(requests[i]));
            MPI_Irecv(cpu_recvbuf + recv_pos,
                    recvcounts[recv_proc],
                    recvtype,
                    recv_proc,
                    tag,
                    comm->global_comm,
                    &(requests[thread_n_msgs + i]));
            idx += num_threads;
        }

        MPI_Waitall(2*thread_n_msgs, requests, MPI_STATUSES_IGNORE);

        free(requests);
    }
} 

/*
    free(ordered_sends);
    free(ordered_recvs);
*/
    ierr += gpuMemcpy(recvbuf, cpu_recvbuf, total_bytes_r, gpuMemcpyHostToDevice);

    cudaFreeHost(cpu_sendbuf);
    cudaFreeHost(cpu_recvbuf);

    return ierr;
}


