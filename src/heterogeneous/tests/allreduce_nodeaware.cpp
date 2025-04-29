#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>


double allreduce_c2mc(int size, float* sendbuf_d, float* sendbuf_h, float* recvbuf_d, float* recvbuf_h,
        float** shared_buffers, float* sendbuf_l, float* recvbuf_l, 
        MPI_Comm comm, MPI_Comm intra_node_comm, MPI_Comm local_comm, MPI_Win& win, int n_iters)
{
    int local_rank, ppn;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &ppn);

    // Get num_gpus per node, set local_gpu
    int num_gpus;
    gpuGetDeviceCount(&num_gpus);
    int ppg = ppn / num_gpus;
    int local_gpu = local_rank / ppg;
    int gpu_rank = local_rank % ppg;

    int size_proc = size / ppn;

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        // Master ranks memcpy from device to host 
        // And copy data into shared buffer
        MPI_Win_fence(0, win);
        if (gpu_rank == 0)
        {
            cudaMemcpy(sendbuf_h, sendbuf_d, size*sizeof(float), cudaMemcpyDeviceToHost);
            memcpy(shared_buffers[local_gpu], sendbuf_h, size*sizeof(float));
        }
        MPI_Win_fence(0, win);

        // Each process accumulates its portion of each buffer
        // (1 buffer per gpu on node)
        for (int i = 0; i < num_gpus; i++)
        {
            float* shared_buf = shared_buffers[i];
            for (int j = 0; j < size_proc; j++)
                sendbuf_l[j] += shared_buf[size_proc*local_rank + j];
        }

        // Inter-node Allreduce (1 proc per node per comm)
        MPI_Allreduce(sendbuf_l, recvbuf_l, size_proc, MPI_FLOAT, MPI_SUM, intra_node_comm);

        // Copy local portion of reduction back to each shared buffer
        for (int i = 0; i < num_gpus; i++)
        {
            float* shared_buf = shared_buffers[i];
            memcpy(&(shared_buf[size_proc*local_rank]), recvbuf_l, size_proc*sizeof(float));
        }
        MPI_Win_fence(0, win);

        // Master ranks memcpy shared buffer back to pinned buffer
        // and finally back to corresponding GPU
        if (gpu_rank == 0)
        {
            memcpy(recvbuf_h, shared_buffers[local_gpu], size*sizeof(float));
            cudaMemcpy(recvbuf_d, recvbuf_h, size*sizeof(float), cudaMemcpyHostToDevice);
        }
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, comm);
    return t0;
}


void allreduce_c2c(int size, float* sendbuf_d, float* sendbuf_h, float* recvbuf_d, float* recvbuf_h,
        MPI_Comm intra_node_comm, int gpu_rank)
{
    if (gpu_rank == 0) 
    {
        cudaMemcpy(sendbuf_h, sendbuf_d, size*sizeof(float), cudaMemcpyDeviceToHost);
        MPI_Allreduce(sendbuf_h, recvbuf_h, size, MPI_FLOAT, MPI_SUM, intra_node_comm);
        cudaMemcpy(recvbuf_d, recvbuf_h, size*sizeof(float), cudaMemcpyHostToDevice);
    }
}

void print_allreduce(int max_p, float* sendbuf_d, float* sendbuf_h, float* recvbuf_d, float* recvbuf_h, 
        float** shared_buffers, float* sendbuf_l, float* recvbuf_l, float* recvbuf_std, float* recvbuf_new,
        MPI_Comm comm, MPI_Comm intra_node_comm, MPI_Comm local_comm, MPI_Win& win)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_rank, ppn;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &ppn);

    // Get num_gpus per node, set local_gpu
    int num_gpus;
    gpuGetDeviceCount(&num_gpus);
    int ppg = ppn / num_gpus;
    int local_gpu = local_rank / ppg;
    int gpu_rank = local_rank % ppg;
    int start_i = log2(ppg);
   
    double max_time;


    for (int i = start_i; i < max_p; i++)
    { 
        int size = pow(2, i);
        int size_proc = size / ppn;

        // Standard C2C Allreduce
        allreduce_c2c(size, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, intra_node_comm, gpu_rank);
        if (gpu_rank == 0)
        {
            cudaMemcpy(recvbuf_std, recvbuf_d, size*sizeof(float), cudaMemcpyDeviceToHost);
        }

        // Copy to Many CPUs Allreduce
        allreduce_c2mc(size, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, shared_buffers, sendbuf_l, recvbuf_l,
                comm, intra_node_comm, local_comm, win, 1);
        if (gpu_rank == 0)
        {
            cudaMemcpy(recvbuf_new, recvbuf_d, size*sizeof(float), cudaMemcpyDeviceToHost);
        }
 
        // Compare solutions for correctness
        if (gpu_rank == 0)
        {
            cudaMemcpy(recvbuf_new, recvbuf_d, size*sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < size; i++)
            {
                if (fabs(recvbuf_std[i] - recvbuf_new[i]) > 1e-06)
                {
                    printf("DIFFERENCE IN RESULTS! %e vs %e\n", recvbuf_std[i], recvbuf_new[i]);
                    MPI_Abort(MPI_COMM_WORLD, -1);
                }
            }  
        }

        // Warm-Up
        double time = allreduce_c2mc(size, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, shared_buffers, sendbuf_l, recvbuf_l,
                comm, intra_node_comm, local_comm, win, 1);

        // Time 2 iterations
        time = allreduce_c2mc(size, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, shared_buffers, sendbuf_l, recvbuf_l,
                comm, intra_node_comm, local_comm, win, 2);
        
        // Calculate iteration count
        MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm);
        int n_iters = (1.0 / time) + 1;

        // Finally, time the c2mc allreduce
        time = allreduce_c2mc(size, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, shared_buffers, sendbuf_l, recvbuf_l,
                comm, intra_node_comm, local_comm, win, n_iters);

        // Print Timing
        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    
        if (rank == 0) printf("Size %d: %e\n", size, max_time);
    }
    if (rank == 0) printf("\n");
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Split MPI_COMM_WORLD into per-node comms
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 
            rank, MPI_INFO_NULL, &local_comm);
    int local_rank, ppn;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &ppn);

    // Get num_gpus per node, set local_gpu
    int num_gpus;
    gpuGetDeviceCount(&num_gpus);
    int ppg = ppn / num_gpus;
    int local_gpu = local_rank / ppg;
    int gpu_rank = local_rank % ppg;
    gpuSetDevice(local_gpu);

    // Split MPI_COMM_WORLD into inter-node comms
    MPI_Comm intra_node_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &intra_node_comm);

    // Max Reduction Size
    int max_p = 31;
    int max_s = pow(2, max_p);
    int max_s_proc = max_s / ppn;

    // Master processes allocate sendbuf and recvbuf
    // Both on master and host
    float *sendbuf_d, *recvbuf_d;
    float *sendbuf_h, *recvbuf_h;
    if (gpu_rank == 0)
    {
        cudaMalloc((void**)&sendbuf_d, max_s*sizeof(float));
        cudaMalloc((void**)&recvbuf_d, max_s*sizeof(float));
        cudaMallocHost((void**)&sendbuf_h, max_s*sizeof(float));
        cudaMallocHost((void**)&recvbuf_h, max_s*sizeof(float));
    }


    // Allocate 1 window
    // All master processes allocate buffers
    MPI_Win win;
    float* baseptr = NULL;
    MPI_Aint winsize = 0;
    if (gpu_rank == 0)
        winsize = max_s*sizeof(float);
    MPI_Win_allocate_shared(winsize, sizeof(float), MPI_INFO_NULL, local_comm, &baseptr, &win);

    // Get shared buffer for each master 
    MPI_Aint size;
    int unit_size;
    void* ptr;
    MPI_Win_shared_query(win, MPI_PROC_NULL, &size, &unit_size, &ptr);
    float* window_base = static_cast<float*>(ptr);

    float** shared_buffers = (float**)malloc(num_gpus*sizeof(float*));
    for (int i = 0; i < num_gpus; i++)
    {
        shared_buffers[i] = window_base + i*max_s;
    }

    // Everyone allocates local buffer
    float *sendbuf_l, *recvbuf_l;
    cudaMallocHost((void**)&sendbuf_l, max_s_proc*sizeof(float));
    cudaMallocHost((void**)&recvbuf_l, max_s_proc*sizeof(float));

    float *recvbuf_std, *recvbuf_new;
    cudaMallocHost((void**)&recvbuf_std, max_s*sizeof(float));
    cudaMallocHost((void**)&recvbuf_new, max_s*sizeof(float));

     
    // Run Main Timing Loops
    print_allreduce(max_p, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, shared_buffers, 
            sendbuf_l, recvbuf_l, recvbuf_std, recvbuf_new, MPI_COMM_WORLD, intra_node_comm,
            local_comm, win);


    // Master process frees send and recv buffers
    if (gpu_rank == 0)
    {
        cudaFree(sendbuf_d);
        cudaFree(recvbuf_d);
        cudaFreeHost(sendbuf_h);
        cudaFreeHost(recvbuf_h);
    }
    // Everyone frees communicators
    MPI_Comm_free(&local_comm);
    MPI_Comm_free(&intra_node_comm);

    MPI_Win_free(&win);
    free(shared_buffers);

    MPI_Finalize();
    return 0;
}
