#include "collective/alltoall.h"
#include "collective/collective.h"

#include "gpu_alltoall.h"

// TODO : SENDCOUNT IS PER MESSAGE ( SO SENDCOUNT * TOTALNUMGPUS)

int cuda_aware_alltoall(int (*f)(const void*, const int, MPI_Datatype, void*, const int, MPI_Dataype, MPI_Comm), 
        const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    return f(sendbuf, sendcount, send_type, recvbuf, recvcount, recvtype, comm->global_comm);
}

int copy_to_cpu_alltoall(int (*f)(const void*, const int, MPI_Datatype, void*, const int, MPI_Dataype, MPI_Comm), 
        const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    gpuSetDevice(comm->rank_gpu);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int n_msgs = comm->num_nodes * comm->gpus_per_node;
    int total_bytes_s = sendcount * send_bytes * n_msgs;
    int total_bytes_r = recvcount * recv_bytes * n_msgs;

    char* cpu_sendbuf = (char*)malloc(total_bytes_s);
    char* cpu_recvbuf = (char*)malloc(total_bytes_r);

    int ierr = 0;

    // Copy from GPU to CPU
    ierr += gpuMemcpyAsync(cpu_sendbuf, sendbuf, total_bytes_s, gpuMemcpyDeviceToHost, comm->proc_stream);
    ierr += gpuStreamSynchronize(comm->proc_stream);

    // Collective Among CPUs
    ierr += f(sendbuf, sendcount, send_type, recvbuf, recvcount, recvtype, comm);

    // Copy from CPU to GPU
    ierr += gpuMemcpyAsync(recvbuf, cpu_recvbuf, total_bytes_r, gpuMemcpyHostToDevice, comm->proc_stream);
    ierr += gpuStreamSynchronize(comm->proc_stream);

    free(cpu_sendbuf);
    free(cpu_recvbuf);
}

int multistep_copy_to_cpu_alltoall(int (*f)(const void*, const int, MPI_Datatype, void*, const int, MPI_Dataype, MPI_Comm), 
        const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        const void* cpu_sendbuf,
        const void* gpu_sendbuf,
        MPIX_Comm* comm)
{
    gpuSetDevice(comm->rank_gpu);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int ierr = 0;
    int n_msgs = comm->num_nodes * comm->gpus_per_node;
    int total_bytes_s = sendcount * send_bytes * n_msgs;
    int total_bytes_r = recvcount * recv_bytes * n_msgs;

    int n_msgs = comm->gpus_per_node * comm->num_nodes;
    int n_msgs_per_gpu = n_msgs / comm->ranks_per_gpu;
    int extra_msgs = n_msgs % comm->ranks_per_gpu;
    int n_msgs_rank = n_msgs_per_gpu;
    if (extra_msgs > comm->gpu_rank) n_msgs_rank++;
    MPI_Status status;

    char* cpu_sendbuf;
    char* cpu_recvbuf;

    if (comm->gpu_rank == 0)
    {
        cpu_sendbuf = (char*)malloc(total_bytes_s);
        cpu_recvbuf = (char*)malloc(total_bytes_r);
    }
    else
    {
        cpu_sendbuf = (char*)malloc(n_msgs_rank * sendcount * send_bytes);
        cpu_recvbuf = (char*)malloc(n_msgs_rank * recvcount * recv_bytes);
    }

    // Copy from GPU to CPU (if gpu_rank == 0)
    if (comm->gpu_rank == 0)
    {
        ierr += gpuMemcpyAsync(cpu_sendbuf, sendbuf, sendcount*send_bytes, gpuMemcpyDeviceToHost, comm->proc_stream);
        ierr += gpuStreamSynchronize(comm->proc_stream);

        // Send data to all other local processes 
        // TODO : could likely be optimized with MPI_Scatterv
        int start = n_msgs_rank;
        for (int i = 1; i < comm->ranks_per_gpu; i++)
        {
            int n_msgs_i = n_msgs_per_gpu;
            if (i < extra_msgs) n_msgs_i++;
            MPI_Send(&(cpu_sendbuf[start]), n_msgs_i * sendcount, sendtype, i, 18765, comm->gpu_comm);
            start += n_msgs_i;
        }
    }
    else
    {
        MPI_Recv(cpu_sendbuf, n_msgs_rank * sendcount, sendtype, 0, 18765, comm->gpu_comm, &status);
    }

    // Collective Among CPUs
    ierr += f(sendbuf, sendcount, send_type, recvbuf, recvcount, recvtype, comm);

    // Copy from CPU to GPU
    if (comm->gpu_rank == 0)
    {
        int start = n_msgs_rank;
        // Recv data from all other local processes
        // TODO : could likely be optimized with MPI_Scatterv
        for (int i = 1; i < comm->ranks_per_gpu; i++)
        {
            int n_msgs_i = n_msgs_per_gpu;
            if (i < extra_msgs) n_msgs_i++;
            MPI_Recv(&(cpu_recvbuf[start]), n_msgs_i * recvcount, recvtype, i, 18766, comm->gpu_comm, &status);
            start += n_msgs_i;
        }
        ierr += gpuMemcpyAsync(recvbuf, cpu_recvbuf, recvcount*recv_bytes, gpuMemcpyHostToDevice, comm->proc_stream);
        ierr += gpuStreamSynchronize(comm->proc_stream);
    }
    else
    {
        MPI_Send(cpu_recvbuf, n_msgs_rank * recvcount, recvtype, 0, 18766, comm->gpu_comm);
    }
}

// TODO : how to share sendbuf and recvbuf (how to use duplicate device pointers??)
int ipc_copy_to_cpu_alltoall(int (*f)(const void*, const int, MPI_Datatype, void*, const int, MPI_Dataype, MPI_Comm), 
        const void* sendbuf, 
        const int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int recvcount, 
        MPI_Datatype recvtype,
        const void* cpu_sendbuf,
        const void* gpu_sendbuf,
        MPIX_Comm* comm)
{
    gpuSetDevice(comm->rank_gpu);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int ierr = 0;
    int n_msgs = comm->gpus_per_node * comm->num_nodes;
    int n_msgs_per_gpu = n_msgs / comm->ranks_per_gpu;
    int extra_msgs = n_msgs % comm->ranks_per_gpu;
    int n_msgs_rank = n_msgs_per_gpu;
    if (extra_msgs > comm->gpu_rank) n_msgs_rank++;
    MPI_Status status;

    int start = n_msgs_per_gpu * rank;
    if (extra < rank) start += extra;
    else start += rank;

    // Share Device Pointer : TODO this should be in an outside function, reused during each alltoall
    if (comm->gpu_rank == 0)
    {
        cudaMalloc(...);
        MPI_Barrier(comm->gpu_comm);
    }
    else
    {
        MPI_Barrier(comm->gpu_comm);
        cudaIpcOpenMemHandle((void **) &d_ptr, s_mem[0].memHandle, cudaIpcMemLazyEnablePeerAccess)
    };

    if (comm->gpu_rank == 0)
    {
        MPI_Barrier(comm->gpu_comm);
    }
    else
    {
        checkCudaErrors(cudaIpcCloseMemHandle(d_ptr));
        MPI_Barrier(comm->gpu_comm);

        // b.3: wait till all the events are used up by proc g_processCount - 1
        procBarrier();

        checkCudaErrors(cudaEventDestroy(event));;

    // Copy from GPU to CPU 
    ierr += gpuMemcpyAsync(cpu_sendbuf, sendbuf + start, sendcount*send_bytes, gpuMemcpyDeviceToHost, comm->proc_stream);
    ierr += gpuStreamSynchronize(comm->proc_stream);

    // Collective Among CPUs
    ierr += f(sendbuf, sendcount, send_type, recvbuf, recvcount, recvtype, comm);

    // Copy from CPU to GPU
    ierr += gpuMemcpyAsync(recvbuf + start, cpu_recvbuf, recvcount*recv_bytes, gpuMemcpyHostToDevice, comm->proc_stream);
    ierr += gpuStreamSynchronize(comm->proc_stream);
}
