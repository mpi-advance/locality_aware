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

#define PPNUMA 72 // number of processes per NUMA region
#define PPS 72  //number of processes per socket
#define PPN 288 //number of processes per node

#define GPNUMA 1 // number of GPUs per NUMA
#define GPS 1 // number of GPUs per socket
#define GPN 4 // number of GPUs per node

typedef void (*pingpong_ftn)(char*, char*, char*, char*, int, int, MPI_Comm, MPI_Request* req);

void ping(char* sendbuf_o, char* sendbuf, char* recvbuf_o, char* recvbuf, 
        int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    MPI_Send(sendbuf, size, MPI_CHAR, proc, 0, comm);
    MPI_Recv(recvbuf, size, MPI_CHAR, proc, 0, comm, MPI_STATUS_IGNORE);
}

void pong(char* sendbuf_o, char* sendbuf, char* recvbuf_o, char* recvbuf, 
        int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    MPI_Recv(recvbuf, size, MPI_CHAR, proc, 0, comm, MPI_STATUS_IGNORE);
    MPI_Send(sendbuf, size, MPI_CHAR, proc, 0, comm);
}

#ifdef GPU
void c2c_ping(char* sendbuf_o, char* sendbuf, char* recvbuf_o, char* recvbuf,
        int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    gpuMemcpy(sendbuf, sendbuf_o, size*sizeof(char), gpuMemcpyDeviceToHost);
    MPI_Send(sendbuf, size, MPI_CHAR, proc, 0, comm);
    MPI_Recv(recvbuf, size, MPI_CHAR, proc, 0, comm, MPI_STATUS_IGNORE);
    gpuMemcpy(recvbuf_o, recvbuf, size*sizeof(char), gpuMemcpyHostToDevice);
}

void c2c_pong(char* sendbuf_o, char* sendbuf, char* recvbuf_o, char* recvbuf,
        int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    MPI_Recv(recvbuf, size, MPI_CHAR, proc, 0, comm, MPI_STATUS_IGNORE);
    gpuMemcpy(recvbuf_o, recvbuf, size*sizeof(char), gpuMemcpyHostToDevice);
    gpuMemcpy(sendbuf, sendbuf_o, size*sizeof(char), gpuMemcpyDeviceToHost);
    MPI_Send(sendbuf, size, MPI_CHAR, proc, 0, comm);
}
#endif

void multi_ping(char* sendbuf_o, char* sendbuf, char* recvbuf_o, char* recvbuf, 
        int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    for (int j = 0; j < size; j++)
        MPI_Isend(&(sendbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);

    for (int j = 0; j < size; j++)
        MPI_Irecv(&(recvbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);
}

void multi_pong(char* sendbuf_o, char* sendbuf, char* recvbuf_o, char* recvbuf, 
        int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    for (int j = 0; j < size; j++)
        MPI_Irecv(&(recvbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);

    for (int j = 0; j < size; j++)
        MPI_Isend(&(sendbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);
}

#ifdef GPU
void c2c_multi_ping(char* sendbuf_o, char* sendbuf, char* recvbuf_o, char* recvbuf, 
        int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    gpuMemcpy(sendbuf, sendbuf_o, size*sizeof(char), gpuMemcpyDeviceToHost);
    for (int j = 0; j < size; j++)
        MPI_Isend(&(sendbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);

    for (int j = 0; j < size; j++)
        MPI_Irecv(&(recvbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);
    gpuMemcpy(recvbuf_o, recvbuf, size*sizeof(char), gpuMemcpyHostToDevice);
}

void c2c_multi_pong(char* sendbuf_o, char* sendbuf, char* recvbuf_o, char* recvbuf, 
        int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    for (int j = 0; j < size; j++)
        MPI_Irecv(&(recvbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);
    gpuMemcpy(recvbuf_o, recvbuf, size*sizeof(char), gpuMemcpyHostToDevice);

    gpuMemcpy(sendbuf, sendbuf_o, size*sizeof(char), gpuMemcpyDeviceToHost);
    for (int j = 0; j < size; j++)
        MPI_Isend(&(sendbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);
}
#endif

void matching_ping(char* sendbuf_o, char* sendbuf, char* recvbuf_o, char* recvbuf, 
        int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    for (int j = 0; j < size; j++)
        MPI_Isend(&(sendbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);

    for (int j = 0; j < size; j++)
        MPI_Irecv(&(recvbuf[j]), 1, MPI_CHAR, proc, size-j-1, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);
}

void matching_pong(char* sendbuf_o, char* sendbuf, char* recvbuf_o, char* recvbuf, 
        int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    for (int j = 0; j < size; j++)
        MPI_Irecv(&(recvbuf[j]), 1, MPI_CHAR, proc, size-j-1, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);

    for (int j = 0; j < size; j++)
        MPI_Isend(&(sendbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);
}

#ifdef GPU
void c2c_matching_ping(char* sendbuf_o, char* sendbuf, char* recvbuf_o, char* recvbuf, 
        int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    gpuMemcpy(sendbuf, sendbuf_o, size*sizeof(char), gpuMemcpyDeviceToHost);
    for (int j = 0; j < size; j++)
        MPI_Isend(&(sendbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);

    for (int j = 0; j < size; j++)
        MPI_Irecv(&(recvbuf[j]), 1, MPI_CHAR, proc, size-j-1, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);
    gpuMemcpy(recvbuf_o, recvbuf, size*sizeof(char), gpuMemcpyHostToDevice);
}

void c2c_matching_pong(char* sendbuf_o, char* sendbuf, char* recvbuf_o, char* recvbuf, 
        int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    for (int j = 0; j < size; j++)
        MPI_Irecv(&(recvbuf[j]), 1, MPI_CHAR, proc, size-j-1, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);
    gpuMemcpy(recvbuf_o, recvbuf, size*sizeof(char), gpuMemcpyHostToDevice);

    gpuMemcpy(sendbuf, sendbuf_o, size*sizeof(char), gpuMemcpyDeviceToHost);
    for (int j = 0; j < size; j++)
        MPI_Isend(&(sendbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);
}
#endif

double ping_pong(pingpong_ftn f_ping, pingpong_ftn f_pong, int rank0, int rank1,
         int size, char* sendbuf_o, char* sendbuf, char* recvbuf_o, char* recvbuf, 
        int n_iters, MPI_Comm comm, MPI_Request* req)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        if (rank == rank0)
        {
            f_ping(sendbuf_o, sendbuf, recvbuf_o, recvbuf, size, rank1, comm, req);
        }
        else if (rank == rank1)
        {
            f_pong(sendbuf_o, sendbuf, recvbuf_o, recvbuf, size, rank0, comm, req);
        }
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    return tfinal;
}

double estimate_iters(pingpong_ftn f_ping, pingpong_ftn f_pong, int rank0, 
        int rank1, int size, char* sendbuf_o, char* sendbuf, char* recvbuf_o, 
        char* recvbuf, MPI_Comm comm, MPI_Request* req)
{
    // Warm-Up
    ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf_o, sendbuf,
            recvbuf_o, recvbuf, 1, comm, req);

    // Time 2 Iterations
    double time = ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf_o, 
            sendbuf, recvbuf_o, recvbuf, 2, comm, req);

    // Get Max Time Across All Procs
    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm);    

    // Set NIters so Timing ~ 1 Second
    int n_iters = (2.0 / time) + 1;
    return n_iters;
}

double time_ping_pong(pingpong_ftn f_ping, pingpong_ftn f_pong, int rank0,
        int rank1, int size, char* sendbuf_o, char* sendbuf, char* recvbuf_o, 
        char* recvbuf, MPI_Comm comm, MPI_Request* req)
{
    int n_iters = estimate_iters(f_ping, f_pong, rank0, rank1, size, sendbuf_o, 
            sendbuf, recvbuf_o, recvbuf, comm, req);
    double time = ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf_o, 
            sendbuf, recvbuf_o, recvbuf, n_iters, comm, req);
    return time;
}


void print_ping_pong(pingpong_ftn f_ping, pingpong_ftn f_pong, int max_p, 
        int rank0, int rank1, char* sendbuf_o, char* sendbuf, char* recvbuf_o, 
        char* recvbuf, MPI_Comm comm, MPI_Request* req)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Print Times
    for (int i = 0; i < max_p; i++)
    {
        int s = pow(2, i);
        if (rank == 0) printf("Size %d: ", s);
        double time = time_ping_pong(f_ping, f_pong, rank0, rank1, s, sendbuf_o, 
                sendbuf, recvbuf_o, recvbuf, comm, req);
        MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (rank == 0) printf("%e\n", time);
    }
    if (rank == 0) printf("\n");
}


void standard_ping_pong(int max_p, char* sendbuf_o, char* sendbuf, 
        char* recvbuf_o, char* recvbuf, MPI_Comm comm, int ppnuma, 
        int pps, int ppn)
{
    int rank, num_procs;
    int rank0, rank1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    pingpong_ftn f_ping = ping;
    pingpong_ftn f_pong = pong;
    if (sendbuf_o != NULL && recvbuf_o != NULL)
    {
        f_ping = c2c_ping;
        f_pong = c2c_pong;
    }

    if (ppnuma != pps && ppnuma > 1)
    {
        if (rank == 0) printf("On-NUMA Ping-Pong:\n");
        rank0 = 0;
        rank1 = ppnuma/2;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_o, sendbuf, 
                recvbuf_o, recvbuf, comm, NULL);
    }

    if (pps > 1)
    {
        if (rank == 0) printf("On-Socket Ping-Pong:\n");
        rank0 = 0;
        rank1 = pps/2;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_o, sendbuf,
                recvbuf_o, recvbuf, comm, NULL);
    }


    if (ppn > 1)
    {
        if (rank == 0) printf("On-Node, Off-Socket Ping-Pong:\n");
        rank0 = 0;
        rank1 = ppn/2;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_o, sendbuf,
                recvbuf_o, recvbuf, comm, NULL);
    }

    if (num_procs > ppn)
    {
        if (rank == 0) printf("Off-Node Ping-Pong:\n");
        rank0 = 0;
        rank1 = ppn;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_o, sendbuf,
                recvbuf_o, recvbuf, comm, NULL);
    }
}

void multiproc_ping_pong_step(int max_p, char* sendbuf_o, char* sendbuf, 
        char* recvbuf_o, char* recvbuf, int max_n, int diff, 
        bool cond0, bool cond1, MPI_Comm comm)
{
    int rank, num_procs;
    int rank0, rank1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    pingpong_ftn f_ping = ping;
    pingpong_ftn f_pong = pong;
    if (sendbuf_o != NULL && recvbuf_o != NULL)
    {
        f_ping = c2c_ping;
        f_pong = c2c_pong;
    }

    rank0 = 0;
    rank1 = diff;
    for (int i = 1; i <= 2*max_n; i*=2)
    {
        // Test largest size in non-power-of-two cases
        if (i > max_n) 
            i = max_n;

        if (rank == 0) printf("Active Procs: %d\n", i);
        if (rank % max_n < i)
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
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_o, sendbuf, 
                recvbuf_o, recvbuf, comm, NULL);

        // Only test i == size one time, then break
        if (i == max_n)
            break;
    }
}

void multiproc_ping_pong(int max_p, char* sendbuf_o, char* sendbuf,
        char* recvbuf_o, char* recvbuf, MPI_Comm comm, int ppnuma, 
        int pps, int ppn)
{
    int rank, num_procs, size;
    bool cond0, cond1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    // On-NUMA, MultiProc (Standard)
    if (ppnuma != pps && ppnuma > 1)
    {
        if (rank == 0) printf("On-NUMA MultiProc Ping-Pong:\n");
        size = ppnuma/2;
        cond0 = rank < size;
        cond1 = rank < 2*size;
        multiproc_ping_pong_step(max_p, sendbuf_o, sendbuf, recvbuf_o, recvbuf, 
                size, size, cond0, cond1, comm);
    }

    // On-Socket, MultiProc (Standard)
    if (pps > 1)
    {
        if (rank == 0) printf("On-Socket MultiProc Ping-Pong:\n");
        size = pps/2;
        cond0 = rank < size;
        cond1 = rank < 2*size;
        multiproc_ping_pong_step(max_p, sendbuf_o, sendbuf, recvbuf_o, recvbuf, 
                size, size, cond0, cond1, comm);
    }

    // On-Node, Off-Socket, MultiProc (Standard)
    if (ppn > 1)
    {
        if (rank == 0) printf("On-Node, Off-Socket MultiProc Ping-Pong:\n");
        size = ppn/2;
        cond0 = rank < size;
        cond1 = rank < 2*size;
        multiproc_ping_pong_step(max_p, sendbuf_o, sendbuf, recvbuf_o, recvbuf, 
                size, size, cond0, cond1, comm);
    }

    // Off-Node, MultiProc (Standard)
    if (num_procs >= 2*ppn)
    {
        if (rank == 0) printf("Off-Node MultiProc Ping-Pong:\n");
        size = ppn;
        cond0 = rank < size;
        cond1 = rank < 2*size;
        multiproc_ping_pong_step(max_p, sendbuf_o, sendbuf, recvbuf_o, recvbuf, 
                size, size, cond0, cond1, comm);
    }

    // On-NUMA, MultiProc (All NUMAs active on 1 Node)
    if (ppnuma != pps && ppnuma > 1)
    {
        if (rank == 0) printf("On-NUMA MultiProc Ping-Pong, All NUMAs Active:\n");
        size = ppnuma/2;
        cond0 = rank < ppn && (rank % ppnuma) < size;
        cond1 = rank < ppn;
        multiproc_ping_pong_step(max_p, sendbuf_o, sendbuf, recvbuf_o, recvbuf, 
                size, size, cond0, cond1, comm);
    }

    // On-Socket, MultiProc (All Sockets active on 1 Node) 
    if (pps > 1)
    {
        if  (rank == 0) printf("On-Socket MultiProc Ping-Pong, All Sockets Active:\n");
        size = pps/2;
        cond0 = rank < ppn && (rank % pps) < size;
        cond1 = rank < ppn;
        multiproc_ping_pong_step(max_p, sendbuf_o, sendbuf, recvbuf_o, recvbuf, 
                size, size, cond0, cond1, comm);
    }

    // Off-Node, MultiProc, Even NUMA Regions
    if (num_procs >= 2*ppn)
    {
        if (rank == 0) printf("Off-Node MultiProc Ping-Pong, Even NUMA Regions:\n");
        size = ppnuma;
        cond0 = rank < ppn;
        cond1 = rank < 2*ppn;
        multiproc_ping_pong_step(max_p, sendbuf_o, sendbuf, recvbuf_o, recvbuf, 
                size, ppn, cond0, cond1, comm);
    }

    // Off-Node, MultiProc, Even Sockets
    if (num_procs >= 2*ppn)
    {
        if (rank == 0) printf("Off-Node MultiProc Ping-Pong, Even Sockets:\n");
        size = pps;
        cond0 = rank < ppn;
        cond1 = rank < 2*ppn;
        multiproc_ping_pong_step(max_p, sendbuf_o, sendbuf, recvbuf_o, recvbuf, 
                size, ppn, cond0, cond1, comm);
    }
}

void multi_ping_pong(int max_p, char* sendbuf_o, char* sendbuf, char* recvbuf_o, 
        char* recvbuf, MPI_Request* req, MPI_Comm comm, int ppnuma, int pps,
        int ppn)
{   
    int rank, num_procs;
    int rank0, rank1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    pingpong_ftn f_ping = multi_ping;
    pingpong_ftn f_pong = multi_pong;
    if (sendbuf_o != NULL && recvbuf_o != NULL)
    {
        f_ping = c2c_multi_ping;
        f_pong = c2c_multi_pong;
    }
    
    if (ppnuma != pps && ppnuma > 1)
    {
        if (rank == 0) printf("On-NUMA Multi Ping-Pong:\n");
        rank0 = 0;
        rank1 = ppnuma/2;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_o, 
                sendbuf, recvbuf_o, recvbuf, comm, req);
    }
    
    if (pps > 1)
    {
        if (rank == 0) printf("On-Socket Multi Ping-Pong:\n");
        rank0 = 0;
        rank1 = pps/2;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_o, 
                sendbuf, recvbuf_o, recvbuf, comm, req);
    }

    if (ppn > 1)
    {
        if (rank == 0) printf("On-Node, Off-Socket Multi Ping-Pong:\n");
        rank0 = 0;
        rank1 = ppn/2;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_o, 
                sendbuf, recvbuf_o, recvbuf, comm, req);
    }

    if (num_procs > ppn)
    {
        if (rank == 0) printf("Off-Node Multi Ping-Pong:\n");
        rank0 = 0;
        rank1 = ppn;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_o, 
                sendbuf, recvbuf_o, recvbuf, comm, req);
    }
}


void matching_ping_pong(int max_p, char* sendbuf_o, char* sendbuf, 
        char* recvbuf_o, char* recvbuf, MPI_Request* req, MPI_Comm comm,
        int ppnuma, int pps, int ppn)
{   
    int rank, num_procs;
    int rank0, rank1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    pingpong_ftn f_ping = matching_ping;
    pingpong_ftn f_pong = matching_pong;
    if (sendbuf_o != NULL && recvbuf_o != NULL)
    {
        f_ping = c2c_matching_ping;
        f_pong = c2c_matching_pong;
    }

    if (ppnuma != pps && pps > 1)
    {
        if (rank == 0) printf("On-NUMA Matching Ping-Pong:\n");
        rank0 = 0;
        rank1 = ppnuma/2;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, 
                sendbuf_o, sendbuf, recvbuf_o, recvbuf, comm, req);
    }

    if (pps > 1)
    {
        if (rank == 0) printf("On-Socket Matching Ping-Pong:\n");
        rank0 = 0;
        rank1 = pps/2;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_o, 
                sendbuf, recvbuf_o, recvbuf, comm, req);
    }

    if (ppn > 1)
    {
        if (rank == 0) printf("On-Node, Off-Socket Matching Ping-Pong:\n");
        rank0 = 0;
        rank1 = ppn/2;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_o,
                sendbuf, recvbuf_o, recvbuf, comm, req);
    }

    if (num_procs > ppn)
    {
        if (rank == 0) printf("Off-Node Matching Ping-Pong:\n");
        rank0 = 0;
        rank1 = ppn;
        print_ping_pong(f_ping, f_pong, max_p, rank0, rank1, sendbuf_o,
                sendbuf, recvbuf_o, recvbuf, comm, req);
    }
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_p = 2;
    int max_n = 2;
    int max_s = pow(2, max_p);
    char* sendbuf = new char[max_s];
    char* recvbuf = new char[max_s];
    MPI_Request* req = new MPI_Request[max_s];

    // Inter-CPU Tests
    if (rank == 0) printf("\n\nInter-CPU Tests:\n\n");
    standard_ping_pong(max_p, NULL, sendbuf, NULL, recvbuf, MPI_COMM_WORLD, 
            PPNUMA, PPS, PPN);

    multiproc_ping_pong(max_p, NULL, sendbuf, NULL, recvbuf, MPI_COMM_WORLD,
            PPNUMA, PPS, PPN);

    multi_ping_pong(max_n, NULL, sendbuf, NULL, recvbuf, req, MPI_COMM_WORLD,
            PPNUMA, PPS, PPN);

    matching_ping_pong(max_n, NULL, sendbuf, NULL, recvbuf, req, MPI_COMM_WORLD,
            PPNUMA, PPS, PPN);    
    delete[] sendbuf;
    delete[] recvbuf;

/*
#ifdef GPU
    // Set Local GPU
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 
            rank, MPI_INFO_NULL, &local_comm);
    int local_rank, ppn;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &ppn);
    MPI_Comm_free(&local_comm);

    int gpn;
    gpuGetDeviceCount(&gpn);

    int ppg = ppn / gpn;
    int local_gpu = rank / ppg;
    int gpu_rank = rank % ppg;

    gpuSetDevice(local_gpu);
    
    MPI_Comm gpu_comm;
    MPI_Comm_split(MPI_COMM_WORLD, gpu_rank, rank, &gpu_comm);

    if (gpu_rank == 0)
    {
        gpuMalloc((void**)&sendbuf, max_s*sizeof(char));
        gpuMalloc((void**)&recvbuf, max_s*sizeof(char));


        char* sendbuf_h;
        char* recvbuf_h;
        gpuMallocHost((void**)&sendbuf_h, max_s*sizeof(char));
        gpuMallocHost((void**)&recvbuf_h, max_s*sizeof(char));


        // GPU-Direct Tests
        if (rank == 0) printf("\n\nGPUDirect Tests:\n\n");
        standard_ping_pong(max_p, NULL, sendbuf, NULL, recvbuf, gpu_comm,
                GPNUMA, GPS, GPN);

        multiproc_ping_pong(max_p, NULL, sendbuf, NULL, recvbuf, gpu_comm,
                GPNUMA, GPS, GPN);

        multi_ping_pong(max_n, NULL, sendbuf, NULL, recvbuf, req, gpu_comm,
                GPNUMA, GPS, GPN);

        matching_ping_pong(max_n, NULL, sendbuf, NULL, recvbuf, req, gpu_comm,
                GPNUMA, GPS, GPN);

        // Copy-To-CPU Tests
        if (rank == 0) printf("\n\nCopy-To-CPU Tests:\n\n");

        standard_ping_pong(max_p, sendbuf, sendbuf_h, recvbuf, recvbuf_h,
                gpu_comm, GPNUMA, GPS, GPN);

        multiproc_ping_pong(max_p, sendbuf, sendbuf_h, recvbuf, recvbuf_h, 
                gpu_comm, GPNUMA, GPS, GPN);

        multi_ping_pong(max_n, sendbuf, sendbuf_h, recvbuf, recvbuf_h, req,
                gpu_comm, GPNUMA, GPS, GPN);

        matching_ping_pong(max_n, sendbuf, sendbuf_h, recvbuf, recvbuf_h, req,
                gpu_comm, GPNUMA, GPS, GPN);


        gpuFreeHost(sendbuf_h);
        gpuFreeHost(recvbuf_h);

        gpuFree(sendbuf);
        gpuFree(recvbuf);
    }

    MPI_Comm_free(&gpu_comm);
#endif
*/
    delete[] req; 

    MPI_Finalize();
    return 0;
}
