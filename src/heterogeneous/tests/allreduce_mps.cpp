#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

void cudaCheck(cudaError_t ierr)
{
    if (ierr != cudaSuccess)
    {
        printf("ERROR! %s\n", cudaGetErrorString(ierr));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

double allreduce(int size, float* sendbuf, float* sendbuf_loc, 
        float* recvbuf, float* recvbuf_loc, MPI_Comm comm, int n_iters)
{
    MPI_Request gpu_req;
    gpuDeviceSynchronize();
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        MPI_Allreduce(sendbuf, recvbuf, size, MPI_FLOAT, MPI_SUM, comm);
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    return tfinal;
}

double allreduce_loc(int size, float* sendbuf, float* recvbuf_1, float* tmpbuf_1,
    float* recvbuf_2, float* tmpbuf_2, float* recvbuf_3,
        int* recvcounts, MPI_Comm comm, MPI_Comm intra_comm, 
        MPI_Comm inter_comm, MPI_Comm socket_comm, int n_iters, int nonFull, int inter_size)
{   
    MPI_Request gpu_req;
    gpuDeviceSynchronize();
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        if (intra_comm != MPI_COMM_NULL)
            MPI_Reduce_scatter(sendbuf, recvbuf_1, recvcounts, MPI_FLOAT,
                    MPI_SUM, intra_comm);
        if (nonFull)
        {
            char dummy = '\0';
            MPI_Bcast((void *)&dummy, 1, MPI_CHAR, 0, socket_comm);
        }
        MPI_Allreduce(recvbuf_2, tmpbuf_1, inter_size, MPI_FLOAT,
                MPI_SUM, inter_comm);
        if (nonFull)
            MPI_Barrier(socket_comm);
        if (intra_comm != MPI_COMM_NULL)
            MPI_Allgather(tmpbuf_2, recvcounts[0], MPI_FLOAT, recvbuf_3, 
                    recvcounts[0], MPI_FLOAT, intra_comm);
        if (nonFull)
        {
            char dummy = '\0';
            MPI_Bcast((void *)&dummy, 1, MPI_CHAR, 0, socket_comm);
        }
        else
            MPI_Barrier(socket_comm);
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    return tfinal;
}

double allreduce_lane(int size, float* sendbuf, float* recvbuf, float* tmpbuf_1, float* tmpbuf_2,
        int* recvcounts, MPI_Comm comm, MPI_Comm intra_comm, 
        MPI_Comm inter_comm, MPI_Comm socket_comm, int n_iters, int nonFull, int inter_size)
{
    MPI_Request gpu_req;
    gpuDeviceSynchronize();
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        MPI_Allreduce(sendbuf, tmpbuf_1, inter_size, MPI_FLOAT,
                MPI_SUM, inter_comm);
        if (nonFull)
            MPI_Barrier(socket_comm);
        if (intra_comm != MPI_COMM_NULL)
            MPI_Allreduce(tmpbuf_2, recvbuf, recvcounts[0], MPI_FLOAT,
                    MPI_SUM, intra_comm);
        MPI_Barrier(socket_comm);
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    return tfinal;
}

void print_allreduce(int max_p, float* sendbuf_d, float* sendbuf_d_local, 
        float* recvbuf_d, float* tmpbuf, float* recvbuf_d_local, float* recvbuf, float* recvbuf_std,
        MPI_Comm comm, MPI_Comm intra_comm, MPI_Comm inter_comm, MPI_Comm socket_comm, int ppg, int socket_rank, int local_gpu)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    double max_time;

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    int* recvcounts_pergpu = new int[num_gpus];
    int* recvcounts_s = new int[num_gpus];
    int* recvcounts_s_pergpu = new int[num_gpus];
    int start_i = log2(ppg * num_gpus);
    for (int i = start_i; i < max_p; i++)
    { 
        double time = 0;
        int s_g = pow(2, i);
        int pergpu = s_g / num_gpus;
        int s = s_g / ppg;
        int s_pergpu = s / num_gpus;
        if (rank == 0) printf("Size %d (Per Proc %d, %d, %d): ", s_g, pergpu, s, s_pergpu);
        
        for (int i = 0; i < num_gpus; i++)
        {
            recvcounts_pergpu[i] = pergpu;
            recvcounts_s[i] = s;
            recvcounts_s_pergpu[i] = s_pergpu;
        }

        // Compare Results
        // 1. Multi-Proc
        allreduce(s, &(sendbuf_d[s*socket_rank]), sendbuf_d_local,
                &(recvbuf_d[s*socket_rank]), recvbuf_d_local,  comm, 1);
        gpuMemcpy(recvbuf_std, &(recvbuf_d[socket_rank*s]), s*sizeof(float), gpuMemcpyDeviceToHost);

        if (socket_rank == 0) 
        {
            MPI_Allreduce(sendbuf_d, recvbuf_d, s_g, MPI_FLOAT, MPI_SUM, comm);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        gpuMemcpy(recvbuf, &(recvbuf_d[socket_rank*s]), s*sizeof(float), gpuMemcpyDeviceToHost);
        for (int i = 0; i < s; i++)
            if (fabs(recvbuf[i] - recvbuf_std[i]) > 1e-6) 
            {
                printf("DIFFERENCE IN RESULTS in STD MPS! %e vs %e\n", recvbuf[i], recvbuf_std[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
        cudaMemset(&(recvbuf_d[socket_rank*s]), 0, s*sizeof(float));
        MPI_Barrier(MPI_COMM_WORLD);
        allreduce_loc(s_g, &(sendbuf_d[socket_rank*s]), &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]),
                &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]),
                &(tmpbuf[socket_rank*s]), &(recvbuf_d[socket_rank*s]), recvcounts_s_pergpu, comm, intra_comm, inter_comm, socket_comm, 1, false, recvcounts_s_pergpu[0]);
        gpuMemcpy(recvbuf_std, &(recvbuf_d[socket_rank*s]), s*sizeof(float), gpuMemcpyDeviceToHost);
        for (int i = 0; i < s; i++)
            if (fabs(recvbuf[i] - recvbuf_std[i]) > 1e-6) 
            {
                printf("DIFFERENCE IN RESULTS in LOC MPS FULL! %e vs %e\n", recvbuf[i], recvbuf_std[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
     
        cudaMemset(&(recvbuf_d[socket_rank*s]), 0, s*sizeof(float));
        MPI_Barrier(MPI_COMM_WORLD);
        allreduce_loc(s_g, sendbuf_d, &(recvbuf_d[local_gpu*pergpu]),
                &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]),
                tmpbuf, recvbuf_d, recvcounts_pergpu, comm, (socket_rank == 0) ? intra_comm : MPI_COMM_NULL, inter_comm, socket_comm, 1, true, recvcounts_s_pergpu[0]);
        gpuMemcpy(recvbuf_std, &(recvbuf_d[socket_rank*s]), s*sizeof(float), gpuMemcpyDeviceToHost);
        for (int i = 0; i < s; i++)
            if (fabs(recvbuf[i] - recvbuf_std[i]) > 1e-6) 
            {
                printf("DIFFERENCE IN RESULTS in LOC MPS! %e vs %e\n", recvbuf[i], recvbuf_std[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
        cudaMemset(&(recvbuf_d[socket_rank*s]), 0, s*sizeof(float));
        MPI_Barrier(MPI_COMM_WORLD);
        allreduce_lane(s_g, &(sendbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                            &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                            &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                            &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), recvcounts_s_pergpu, comm, intra_comm, inter_comm, socket_comm, 1, false, recvcounts_s_pergpu[0]);
        gpuMemcpy(recvbuf_std, &(recvbuf_d[socket_rank*s]), s*sizeof(float), gpuMemcpyDeviceToHost);
        for (int i = 0; i < s; i++)
            if (fabs(recvbuf[i] - recvbuf_std[i]) > 1e-6) 
            {
                printf("DIFFERENCE IN RESULTS in LANE MPS FULL! %e vs %e\n", recvbuf[i], recvbuf_std[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
     
        cudaMemset(&(recvbuf_d[socket_rank*s]), 0, s*sizeof(float));
        MPI_Barrier(MPI_COMM_WORLD);
        allreduce_lane(s_g, &(sendbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                            &(recvbuf_d[local_gpu*pergpu]), 
                            &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                            &(tmpbuf[local_gpu*pergpu]), recvcounts_pergpu, comm, (socket_rank == 0) ? intra_comm : MPI_COMM_NULL, inter_comm, socket_comm, 1, true, recvcounts_s_pergpu[0]);
        gpuMemcpy(recvbuf_std, &(recvbuf_d[socket_rank*s]), s*sizeof(float), gpuMemcpyDeviceToHost);
        for (int i = 0; i < s; i++)
            if (fabs(recvbuf[i] - recvbuf_std[i]) > 1e-6) 
            {
                printf("DIFFERENCE IN RESULTS in LANE MPS! %e vs %e\n", recvbuf[i], recvbuf_std[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
        // Warm-Up
        allreduce(s, &(sendbuf_d[s*socket_rank]), sendbuf_d_local,
                &(recvbuf_d[s*socket_rank]), recvbuf_d_local,  comm, 1);

        // Time 2 iterations
        time = allreduce(s, &(sendbuf_d[s*socket_rank]), sendbuf_d_local,
                &(recvbuf_d[s*socket_rank]), recvbuf_d_local,  comm, 2);

        // Get Max Time
        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    

        // Set NIters so Timing ~ 1 Second
        int n_iters = (2.0 / max_time) + 1;

        // Time Allreduce
        time = allreduce(s, &(sendbuf_d[s*socket_rank]), sendbuf_d_local,
                &(recvbuf_d[s*socket_rank]), recvbuf_d_local,  comm, n_iters);

        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    
        if (rank == 0) printf("STD MPS: %e\n", max_time);
        
        // Warm-Up
        allreduce_loc(s_g, &(sendbuf_d[socket_rank*s]), &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]),
                &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]),
                &(tmpbuf[socket_rank*s]), &(recvbuf_d[socket_rank*s]), recvcounts_s_pergpu, comm, intra_comm, inter_comm, socket_comm, 1, false, recvcounts_s_pergpu[0]);

        // Time 2 iterations
        time = allreduce_loc(s_g, &(sendbuf_d[socket_rank*s]), &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]),
                &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]),
                &(tmpbuf[socket_rank*s]), &(recvbuf_d[socket_rank*s]), recvcounts_s_pergpu, comm, intra_comm, inter_comm, socket_comm, 2, false, recvcounts_s_pergpu[0]);

        // Get Max Time
        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    

        // Set NIters so Timing ~ 1 Second
        n_iters = (2.0 / max_time) + 1;

        // Time Allreduce
        time = allreduce_loc(s_g, &(sendbuf_d[socket_rank*s]), &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]),
                &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]),
                &(tmpbuf[socket_rank*s]), &(recvbuf_d[socket_rank*s]), recvcounts_s_pergpu, comm, intra_comm, inter_comm, socket_comm, n_iters, false, recvcounts_s_pergpu[0]);

        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    
        if (rank == 0) printf("LOC MPS FULL: %e\n", max_time);
        
        // Warm-Up
        allreduce_loc(s_g, sendbuf_d, &(recvbuf_d[local_gpu*pergpu]),
                &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]),
                tmpbuf, recvbuf_d, recvcounts_pergpu, comm, (socket_rank == 0) ? intra_comm : MPI_COMM_NULL, inter_comm, socket_comm, 1, true, recvcounts_s_pergpu[0]);

        // Time 2 iterations
        time = allreduce_loc(s_g, sendbuf_d, &(recvbuf_d[local_gpu*pergpu]),
                &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]),
                tmpbuf, recvbuf_d, recvcounts_pergpu, comm, (socket_rank == 0) ? intra_comm : MPI_COMM_NULL, inter_comm, socket_comm, 2, true, recvcounts_s_pergpu[0]);

        // Get Max Time
        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    

        // Set NIters so Timing ~ 1 Second
        n_iters = (2.0 / max_time) + 1;

        // Time Allreduce
        time = allreduce_loc(s_g, sendbuf_d, &(recvbuf_d[local_gpu*pergpu]),
                &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]),
                tmpbuf, recvbuf_d, recvcounts_pergpu, comm, (socket_rank == 0) ? intra_comm : MPI_COMM_NULL, inter_comm, socket_comm, n_iters, true, recvcounts_s_pergpu[0]);

        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    
        if (rank == 0) printf("LOC MPS: %e\n", max_time);
        
        // Warm-Up
        allreduce_lane(s_g, &(sendbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                    &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                    &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                    &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), recvcounts_s_pergpu, comm, intra_comm, inter_comm, socket_comm, 1, false, recvcounts_s_pergpu[0]);

        // Time 2 iterations
        time = allreduce_lane(s_g, &(sendbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                    &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                    &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                    &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), recvcounts_s_pergpu, comm, intra_comm, inter_comm, socket_comm, 2, false, recvcounts_s_pergpu[0]);

        // Get Max Time
        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    

        // Set NIters so Timing ~ 1 Second
        n_iters = (2.0 / max_time) + 1;

        // Time Allreduce
        time = allreduce_lane(s_g, &(sendbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                    &(recvbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                    &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                    &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), recvcounts_s_pergpu, comm, intra_comm, inter_comm, socket_comm, n_iters, false, recvcounts_s_pergpu[0]);

        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    
        if (rank == 0) printf("LANE MPS FULL: %e\n", max_time);
        
        // Warm-Up
        allreduce_lane(s_g, &(sendbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                            &(recvbuf_d[local_gpu*pergpu]), 
                            &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                            &(tmpbuf[local_gpu*pergpu]), recvcounts_pergpu, comm, (socket_rank == 0) ? intra_comm : MPI_COMM_NULL, inter_comm, socket_comm, 1, true, recvcounts_s_pergpu[0]);

        // Time 2 iterations
        time = allreduce_lane(s_g, &(sendbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                            &(recvbuf_d[local_gpu*pergpu]), 
                            &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                            &(tmpbuf[local_gpu*pergpu]), recvcounts_pergpu, comm, (socket_rank == 0) ? intra_comm : MPI_COMM_NULL, inter_comm, socket_comm, 2, true, recvcounts_s_pergpu[0]);

        // Get Max Time
        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    

        // Set NIters so Timing ~ 1 Second
        n_iters = (2.0 / max_time) + 1;

        // Time Allreduce
        time = allreduce_lane(s_g, &(sendbuf_d[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                            &(recvbuf_d[local_gpu*pergpu]), 
                            &(tmpbuf[(socket_rank*s) + (local_gpu*s_pergpu)]), 
                            &(tmpbuf[local_gpu*pergpu]), recvcounts_pergpu, comm, (socket_rank == 0) ? intra_comm : MPI_COMM_NULL, inter_comm, socket_comm, n_iters, true, recvcounts_s_pergpu[0]);

        MPI_Allreduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    
        if (rank == 0) printf("LANE MPS: %e\n", max_time);
        
        fflush(stdout);
    }
    if (rank == 0) printf("\n");
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Set Local GPU
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 
            rank, MPI_INFO_NULL, &local_comm);
    int local_rank, ppn;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &ppn);

    int gpn;
    cudaCheck(cudaGetDeviceCount(&gpn));

    int ppg = ppn / gpn;
    int local_gpu = local_rank / ppg;
    int gpu_rank = local_rank % ppg;

    int max_p = 30;
    int max_s = pow(2, max_p);
    int max_s_proc = max_s / ppg;

    if (argc < 2 || (strcmp(argv[1], "r") != 0))
    {
        cudaCheck(cudaSetDevice(local_gpu));
    }
    else
    {
        cudaSetDevice((gpn - local_gpu) - 1);
        printf("rank %d reversed\n", rank);
        fflush(stdout);
    }
    
    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);
    
    MPI_Comm lane_comm;
    MPI_Comm_split(MPI_COMM_WORLD, local_rank, rank / ppn, &lane_comm);

    MPI_Comm socket_comm;
    MPI_Comm_split(local_comm, local_gpu, rank, &socket_comm);
    
    MPI_Comm intra_comm;
    MPI_Comm_split(local_comm, local_rank % ppg, local_rank / ppg, &intra_comm);

    float* sendbuf_d;
    float* recvbuf_d;
    float* tmpbuf_d;
    float* sendbuf;
    float* recvbuf;
    float* recvbuf_std;
    float* sendbuf_d_local;
    float* recvbuf_d_local;

    cudaCheck(cudaMallocHost((void**)&sendbuf, max_s_proc*sizeof(float)));
    cudaCheck(cudaMallocHost((void**)&recvbuf, max_s_proc*sizeof(float)));
    cudaCheck(cudaMallocHost((void**)&recvbuf_std, max_s_proc*sizeof(float)));

    cudaCheck(cudaMalloc((void**)&sendbuf_d_local, max_s_proc*sizeof(float)));
    cudaCheck(cudaMalloc((void**)&recvbuf_d_local, max_s_proc*sizeof(float)));

    cudaIpcMemHandle_t send_handle, recv_handle, tmp_handle;
    if (gpu_rank == 0)
    {
        cudaCheck(cudaMalloc((void**)&sendbuf_d, max_s*sizeof(float)));
        cudaCheck(cudaMalloc((void**)&recvbuf_d, max_s*sizeof(float)));
        cudaCheck(cudaMalloc((void**)&tmpbuf_d, max_s*sizeof(float)));

        cudaCheck(cudaIpcGetMemHandle(&send_handle, sendbuf_d));
        cudaCheck(cudaIpcGetMemHandle(&recv_handle, recvbuf_d));
        cudaCheck(cudaIpcGetMemHandle(&tmp_handle, tmpbuf_d));
    }
    MPI_Bcast(&send_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, socket_comm);
    MPI_Bcast(&recv_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, socket_comm);
    MPI_Bcast(&tmp_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, socket_comm);
    if (gpu_rank != 0)
    {
        cudaCheck(cudaIpcOpenMemHandle((void**)&sendbuf_d, send_handle, cudaIpcMemLazyEnablePeerAccess)); 
        cudaCheck(cudaIpcOpenMemHandle((void**)&recvbuf_d, recv_handle, cudaIpcMemLazyEnablePeerAccess)); 
        cudaCheck(cudaIpcOpenMemHandle((void**)&tmpbuf_d, tmp_handle, cudaIpcMemLazyEnablePeerAccess)); 
    }

    for (int i = 0; i < max_s_proc; i++)
        sendbuf[i] = ((float)rand()) / RAND_MAX;
    cudaCheck(cudaMemcpy(&(sendbuf_d[gpu_rank*max_s_proc]), sendbuf, max_s_proc*sizeof(float), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    cudaCheck(cudaDeviceSynchronize());

    if (rank == 0) printf("Starting Allreduce Timings, PPG %d:\n", ppg);
    print_allreduce(max_p, sendbuf_d, sendbuf_d_local, recvbuf_d, tmpbuf_d, recvbuf_d_local, 
            recvbuf, recvbuf_std, gpu_comm, intra_comm, lane_comm, socket_comm, ppg, gpu_rank, local_gpu);
    MPI_Barrier(MPI_COMM_WORLD);

    if (gpu_rank == 0)
    {
        cudaCheck(cudaFree(sendbuf_d));
        cudaCheck(cudaFree(recvbuf_d));
        cudaCheck(cudaFree(tmpbuf_d));
    }
    else
    {
        cudaCheck(cudaIpcCloseMemHandle(sendbuf_d));
        cudaCheck(cudaIpcCloseMemHandle(recvbuf_d));
        cudaCheck(cudaIpcCloseMemHandle(tmpbuf_d));
    }

    cudaCheck(cudaFree(sendbuf_d_local));
    cudaCheck(cudaFree(recvbuf_d_local));

    cudaCheck(cudaFreeHost(sendbuf));
    cudaCheck(cudaFreeHost(recvbuf));
    cudaCheck(cudaFreeHost(recvbuf_std));

    MPI_Comm_free(&socket_comm);
    MPI_Comm_free(&gpu_comm);
    MPI_Comm_free(&lane_comm);
    MPI_Comm_free(&intra_comm);
    MPI_Comm_free(&local_comm);

    MPI_Finalize();
    return 0;
}
