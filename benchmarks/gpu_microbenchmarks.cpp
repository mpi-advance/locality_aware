#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>


#define NODES 2 //number of nodes
#define SPN 4   //number of sockets per node
#define PPNUMA 4 // number of processes per NUMA region
#define PPS 16  //number of processes per socket
#define PPN 64 //number of processes per node

#define GPNUMA 1
#define GPS 1
#define GPN 4
#define PPG 16

#define MATCHING 0

// Want to store my device ID globally
int device;

typedef void (*pingpong_ftn)(char*, char*, char*, char*, int, int, gpuStream_t stream, MPI_Comm);

void ping_gpu_aware(char* sendbuf_d, char* sendbuf_h, char* recvbuf_d, char* recvbuf_h, int size, int proc,
        gpuStream_t stream, MPI_Comm comm)
{
    MPI_Send(sendbuf_d, size, MPI_CHAR, proc, 0, comm);
    MPI_Recv(recvbuf_d, size, MPI_CHAR, proc, 0, comm, MPI_STATUS_IGNORE);
}

void pong_gpu_aware(char* sendbuf_d, char* sendbuf_h, char* recvbuf_d, char* recvbuf_h, int size, int proc,
        gpuStream_t stream, MPI_Comm comm)
{
    MPI_Recv(recvbuf_d, size, MPI_CHAR, proc, 0, comm, MPI_STATUS_IGNORE);
    MPI_Send(sendbuf_d, size, MPI_CHAR, proc, 0, comm);
}

void ping_copy_to_cpu(char* sendbuf_d, char* sendbuf_h, char* recvbuf_d, char* recvbuf_h, int size, int proc,
        gpuStream_t stream, MPI_Comm comm)
{
//    gpuMemcpyAsync(sendbuf_h, sendbuf_d, size*sizeof(char), gpuMemcpyDeviceToHost, stream);
//    gpuStreamSynchronize(stream);
    gpuMemcpy(sendbuf_h, sendbuf_d, size*sizeof(char), gpuMemcpyDeviceToHost);
    MPI_Send(sendbuf_h, size, MPI_CHAR, proc, 0, comm);
    MPI_Recv(recvbuf_h, size, MPI_CHAR, proc, 0, comm, MPI_STATUS_IGNORE);
    gpuMemcpy(recvbuf_d, recvbuf_h, size*sizeof(char), gpuMemcpyHostToDevice);
//    gpuMemcpyAsync(recvbuf_d, recvbuf_h, size*sizeof(char), gpuMemcpyHostToDevice, stream);
//    gpuStreamSynchronize(stream);
}
void pong_copy_to_cpu(char* sendbuf_d, char* sendbuf_h, char* recvbuf_d, char* recvbuf_h, int size, int proc,
        gpuStream_t stream, MPI_Comm comm)
{
    MPI_Recv(recvbuf_h, size, MPI_CHAR, proc, 0, comm, MPI_STATUS_IGNORE);
    //gpuMemcpyAsync(recvbuf_d, recvbuf_h, size*sizeof(char), gpuMemcpyHostToDevice, stream);
    //gpuStreamSynchronize(stream);
    gpuMemcpy(recvbuf_d, recvbuf_h, size*sizeof(char), gpuMemcpyHostToDevice);
    gpuMemcpy(sendbuf_h, sendbuf_d, size*sizeof(char), gpuMemcpyDeviceToHost);
    //gpuMemcpyAsync(sendbuf_h, sendbuf_d, size*sizeof(char), gpuMemcpyDeviceToHost, stream);
    //gpuStreamSynchronize(stream);
    MPI_Send(sendbuf_h, size, MPI_CHAR, proc, 0, comm);
}

double ping_pong(pingpong_ftn f_ping, pingpong_ftn f_pong, int rank0, int rank1, int size, char* sendbuf_d, char* sendbuf_h, char* recvbuf_d, char* recvbuf_h, int n_iters, gpuStream_t stream, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    gpuStreamSynchronize(stream);
    gpuDeviceSynchronize();
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        if (rank == rank0)
        {
            f_ping(sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, size, rank1, stream, comm);
        }
        else if (rank == rank1)
        {
            f_pong(sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, size, rank0, stream, comm);
        }
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, comm);
    return t0;
}

int estimate_iters(pingpong_ftn f_ping, pingpong_ftn f_pong, int rank0, int rank1, int size, char* sendbuf_d, 
        char* sendbuf_h, char* recvbuf_d, char* recvbuf_h, gpuStream_t stream, MPI_Comm comm)
{
    double time;

    // Time a single iteration
    time = ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, 
            1, stream, comm);
  
    // If single iteration takes longer than 1 second, we only want 1 iteration
    if (time > 1.0)
        return 1.0;

    // If we will need less than 10 iterations, use 2 iterations to estimate iteration count
    if (time > 1e-01)
    {
        time = ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, 
            2, stream, comm);
    }

    // Otherwise, If we want less than 100 iterations, use 10 iterations to estimate iteration count
    else if (time > 1e-02)
    {
        time = ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, 
            10, stream, comm);
    }

    // Otherwise, use 100 iterations to estimate iteration count
    else
    {
        time = ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, 
            100, stream, comm);
    }

    int n_iters = (1.0 / time) + 1;
    if (n_iters < 1) n_iters = 1;

    return n_iters;
}

double time_ping_pong(pingpong_ftn f_ping, pingpong_ftn f_pong, int rank0, int rank1, int size, char* sendbuf_d, 
        char* sendbuf_h, char* recvbuf_d, char* recvbuf_h, gpuStream_t stream, MPI_Comm comm)
{
    // Warm-Up
    ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, 1, stream, comm);

    // Estimate Number of Iterations for timer precision ~ 1 second
    int n_iters = estimate_iters(f_ping, f_pong, rank0, rank1, size, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, 
            stream, comm);

    double time = ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, 
            n_iters, stream, comm);
    return time;
}


void print_ping_pong(pingpong_ftn f_ping, pingpong_ftn f_pong, int max_p, int rank0, int rank1, char* sendbuf_d,
        char* sendbuf_h, char* recvbuf_d, char* recvbuf_h, gpuStream_t stream, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Print Times
    for (int i = 0; i < max_p; i++)
    {
        //int s = pow(2, i);
        int s = 1 << i;
        if (rank == 0) printf("Size %d: ", s);
        double time = time_ping_pong(f_ping, f_pong, rank0, rank1, s, sendbuf_d, sendbuf_h, 
                recvbuf_d, recvbuf_h, stream, comm);

        // Time has already been reduced to find max across processes
        if (rank == 0) printf("%e\n", time);
    }
    if (rank == 0) printf("\n");
}

void standard_ping_pong_gpu(pingpong_ftn f_ping, pingpong_ftn f_pong, int max_p, char* sendbuf_d, char* sendbuf_h, char* recvbuf_d, char* recvbuf_h,
        gpuStream_t stream, MPI_Comm comm)
{   
    int rank, rank0, rank1;
    MPI_Comm_rank(comm, &rank);
    
    if (GPNUMA <= 1)
    {   
        if (rank == 0) printf("Not measuring GPU-Aware intra-NUMA, because GPNUMA is set to %d\n", GPNUMA);
    }
    else if (GPNUMA == GPN)
    {   
        if (rank == 0) printf("Not measuring GPU-Aware intra-NUMA, because only 1 NUMA per node\n");
    }
    else
    {
        if (rank == 0) printf("GPU-Aware On-NUMA Ping-Pong:\n");
        rank0 = 0;
        rank1 = (GPNUMA*PPG)/2;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_d, sendbuf_h, recvbuf_d,
                recvbuf_h, stream, comm);
    }

    if (GPS <= 1)
    {   
        if (rank == 0) printf("Not measuring GPU-Aware intra-socket, because GPS is set to %d\n", GPS);
    }
    else if (GPS == GPN)
    {   
        if (rank == 0) printf("Not measuring GPU-Aware intra-socket, because only 1 socket per node\n");
    }
    else if (GPNUMA == GPS)
    {
        if (rank == 0) printf("Not measuring GPU-Aware intra-socket because same as intra-NUMA\n");
    }
    else
    {
        if (rank == 0) printf("GPU-Aware On-Socket Ping-Pong:\n");
        rank0 = 0;
        rank1 = (GPS*PPG)/2;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_d, sendbuf_h, recvbuf_d,
                recvbuf_h, stream, comm);
    }


    if (GPN <= 1)
    {   
        if (rank == 0) printf("Not measuring GPU-Aware intra-node, because GPN is set to %d\n", GPN);
    }
    else
    {
        if (rank == 0) printf("GPU-Aware On-Node, Off-Socket Ping-Pong:\n");
        rank0 = 0;
        rank1 = (GPN*PPG)/2;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_d, sendbuf_h, recvbuf_d,
                recvbuf_h, stream, comm);
    }

    if (NODES < 2)
    {   
        printf("Not measuring GPU-Aware intra-node because NODES is set to %d\n", NODES);
    }
    else
    {   
        if (rank == 0) printf("GPU-Aware Off-Node Ping-Pong:\n");
        rank0 = 0;
        rank1 = GPN*PPG;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_d, sendbuf_h, recvbuf_d,
                recvbuf_h, stream, comm);
    }
}

void multiproc_ping_pong_step(pingpong_ftn f_ping, pingpong_ftn f_pong, int max_p, char* sendbuf_d, char* sendbuf_h, char* recvbuf_d, char* recvbuf_h, 
            int max_n, int diff, int div, bool cond0, bool cond1, gpuStream_t stream, MPI_Comm comm)
{
    int rank, rank0, rank1;
    MPI_Comm_rank(comm, &rank);

    rank0 = 0;
    rank1 = diff;
    for (int i = 1; i <= 2*max_n; i*=2)
    {
        // Test largest size in non-power-of-two cases
        if (i > max_n) 
            i = max_n;

        if (rank == 0) printf("Active Procs: %d\n", i);
        if (rank/div % max_n < i)
        {
            if (cond0)
            {
                rank0 = rank; 
                rank1 = rank + diff;
            }
            else if (cond1)
            {
                rank0 = rank - diff;
                rank1 = rank;
            }
        }

        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h,
                stream, comm);

        // Only test i == size one time, then break
        if (i == max_n)
            break;
    }
}

void multiproc_ping_pong_gpu(pingpong_ftn f_ping, pingpong_ftn f_pong, int max_p, char* sendbuf_d, char* sendbuf_h, char* recvbuf_d, char* recvbuf_h,
        gpuStream_t stream, MPI_Comm comm)
{
    int rank, size;
    bool cond0, cond1;
    MPI_Comm_rank(comm, &rank);

    // Off-Node, Multiple Processes Per GPU
    if (PPG <= 1)
    {
        if (rank == 0) printf("Not measuring multiple processes per GPU, because PPG is set to %d\n", PPG);
    }
    else if (NODES < 2)
    {
        if (rank == 0) printf("Not measuring multiple processes per GPU, because NODES is set to %d\n", NODES);
    }
    else
    {
        if (rank == 0) printf("Off-Node Multiple Processes Per GPU:\n");
        size = PPG;
        cond0 = rank < PPG;
        cond1 = rank > PPN && rank < PPN + PPG;
        multiproc_ping_pong_step(f_ping, f_pong, max_p, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, size, PPN, 1, cond0, cond1,
                stream, comm);
    }

    // Off-Node, Multiple Processes Per GPU, All GPUs
    if (PPG <= 1)
    {
        if (rank == 0) printf("Not measuring multiple processes per GPU (all GPUs active), because PPG is set to %d\n", PPG);
    }
    else if (NODES < 2)
    {
        if (rank == 0) printf("Not measuring multiple processes per GPU (all GPUs active), because NODES is set to %d\n", NODES);
    }
    else
    {
        if (rank == 0) printf("Off-Node Multiple Processes Per GPU (all GPUs active per node):\n");
        size = PPG;
        cond0 = rank < PPN;
        cond1 = rank < 2*PPN;
        multiproc_ping_pong_step(f_ping, f_pong, max_p, sendbuf_d, sendbuf_h, recvbuf_d, recvbuf_h, size, PPN, 1, cond0, cond1,
                stream, comm);
    }
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_p = 25;
    //int max_s = pow(2, max_p);
    int max_s = 1 << max_p;

    if (rank == 0) 
    {
        printf("Inter-CPU Microbenchmarks will be run with the following:\n");
        printf("# Processes per NUMA: %d\n", PPNUMA);
        printf("# Processes per socket: %d\n", PPS);
        printf("# Processes per node: %d\n", PPN);
        printf("# of nodes: %d\n", NODES); 
        printf("Inter-GPU Microbenchmarks will be run with the following:\n");
        printf("# GPUs per NUMA: %d\n", GPNUMA);
        printf("# GPUs per socket: %d\n", GPS);
        printf("# GPUs per node: %d\n", GPN);
        printf("# Processes per GPU: %d\n", PPG);
        printf("If these are incorrect, edit the defines at the top of benchmarks/microbenchmarks.cpp\n");
    }

    gpuSetDevice(0);

    char *sendbuf_d, *recvbuf_d;
    char *sendbuf_h, *recvbuf_h;

    gpuMalloc((void**)&sendbuf_d, max_s*sizeof(char));
    gpuMalloc((void**)&recvbuf_d, max_s*sizeof(char));

    gpuMallocHost((void**)&sendbuf_h, max_s*sizeof(char));
    gpuMallocHost((void**)&recvbuf_h, max_s*sizeof(char));

    gpuStream_t stream;
    gpuStreamCreate(&stream);

    if (rank == 0) 
    {
        if (GPNUMA <= 1)
        {
            printf("GPU-Aware intra-NUMA tests will not be run because GPNUMA is set to %d.\n", GPNUMA);
        }
        else if (GPNUMA == GPN)
        {
            printf("GPU-Aware intra-NUMA tests will not be run because there is only one NUMA per node.  The timings will be gathered in the intra-node tests.\n");
        }

        if (GPS <= 1)
        {
            printf("intra-socket tests will not be run because GPS is set to %d.\n", GPS);
        }
        else if (GPS == GPN)
        {
            printf("intra-socket tests will not be run because there is only one socket per node.  The timings will be gathered in inter-node tests.\n");
        }
        else if (GPS == GPNUMA)
        {
            printf("intra-socket tests will not be run because there is only one NUMA per socket.  The timings will be gathered in inter-NUMA tests.\n");
        }

        if (GPN <= 1)
        {
            printf("intra-node tests will not be run because GPN is set to %d\n", GPN);
        }
    }

    gpuGetDevice(&device);

    if (rank == 0) printf("Running standard GPU-Aware benchmarks\n");
    standard_ping_pong_gpu(ping_gpu_aware, pong_gpu_aware, max_p, sendbuf_d, NULL,
            recvbuf_d, NULL, stream, MPI_COMM_WORLD);

    if (rank == 0) printf("Running standard inter-CPU benchmarks on host buffers\n");
    standard_ping_pong_gpu(ping_gpu_aware, pong_gpu_aware, max_p, sendbuf_h, NULL,
            recvbuf_h, NULL, stream, MPI_COMM_WORLD);

    if (rank == 0) printf("Running standard Copy-to-CPU benchmarks\n");
    standard_ping_pong_gpu(ping_copy_to_cpu, pong_copy_to_cpu, max_p, sendbuf_d, sendbuf_h,
            recvbuf_d, recvbuf_h, stream, MPI_COMM_WORLD);

    if (rank == 0) printf("Running multi-proc GPU-Aware benchmarks\n");
    multiproc_ping_pong_gpu(ping_gpu_aware, pong_gpu_aware, max_p, sendbuf_d, sendbuf_h,
            recvbuf_d, recvbuf_h, stream, MPI_COMM_WORLD);

    if (rank == 0) printf("Running multi-proc Copy-to-CPU benchmarks\n");
    multiproc_ping_pong_gpu(ping_copy_to_cpu, pong_copy_to_cpu, max_p, sendbuf_d, sendbuf_h,
            recvbuf_d, recvbuf_h, stream, MPI_COMM_WORLD);

    gpuStreamDestroy(stream);

    gpuFreeHost(sendbuf_h);
    gpuFreeHost(recvbuf_h);

    gpuFree(sendbuf_d);
    gpuFree(recvbuf_d);

    MPI_Finalize();
    return 0;
}
