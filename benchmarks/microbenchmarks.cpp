#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>

#define NODES 2 //number of nodes
#define SPN 2   //number of sockets per node
#define PPNUMA 14 // number of processes per NUMA region
#define PPS 56  //number of processes per socket
#define PPN 112 //number of processes per node

typedef void (*pingpong_ftn)(char*, char*, int, int, MPI_Comm, MPI_Request* req);

void ping(char* sendbuf, char* recvbuf, int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    MPI_Send(sendbuf, size, MPI_CHAR, proc, 0, comm);
    MPI_Recv(recvbuf, size, MPI_CHAR, proc, 0, comm, MPI_STATUS_IGNORE);
}

void pong(char* sendbuf, char* recvbuf, int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    MPI_Recv(recvbuf, size, MPI_CHAR, proc, 0, comm, MPI_STATUS_IGNORE);
    MPI_Send(sendbuf, size, MPI_CHAR, proc, 0, comm);
}

void multi_ping(char* sendbuf, char* recvbuf, int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    for (int j = 0; j < size; j++)
        MPI_Isend(&(sendbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);

    for (int j = 0; j < size; j++)
        MPI_Irecv(&(recvbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);
}

void multi_pong(char* sendbuf, char* recvbuf, int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    for (int j = 0; j < size; j++)
        MPI_Irecv(&(recvbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);

    for (int j = 0; j < size; j++)
        MPI_Isend(&(sendbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);
}

void matching_ping(char* sendbuf, char* recvbuf, int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    for (int j = 0; j < size; j++)
        MPI_Isend(&(sendbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);

    for (int j = 0; j < size; j++)
        MPI_Irecv(&(recvbuf[j]), 1, MPI_CHAR, proc, size-j-1, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);
}

void matching_pong(char* sendbuf, char* recvbuf, int size, int proc, MPI_Comm comm, MPI_Request* req)
{
    for (int j = 0; j < size; j++)
        MPI_Irecv(&(recvbuf[j]), 1, MPI_CHAR, proc, size-j-1, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);

    for (int j = 0; j < size; j++)
        MPI_Isend(&(sendbuf[j]), 1, MPI_CHAR, proc, j, comm, &(req[j]));
    MPI_Waitall(size, req, MPI_STATUSES_IGNORE);
}

double ping_pong(pingpong_ftn f_ping, pingpong_ftn f_pong, int rank0, int rank1, int size, char* sendbuf, char* recvbuf, int n_iters, 
        MPI_Comm comm, MPI_Request* req)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        if (rank == rank0)
        {
            f_ping(sendbuf, recvbuf, size, rank1, comm, req);
        }
        else if (rank == rank1)
        {
            f_pong(sendbuf, recvbuf, size, rank0, comm, req);
        }
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    return tfinal;
}

double estimate_iters(pingpong_ftn f_ping, pingpong_ftn f_pong, int rank0, int rank1, int size, char* sendbuf, 
        char* recvbuf, MPI_Comm comm, MPI_Request* req)
{
    double time = ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf, recvbuf, 5, comm, req);
    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm);    
    int n_iters = (1.0 / time) + 1;
    if (time > 1.0)
        n_iters = 1;
    return n_iters;
}

double time_ping_pong(pingpong_ftn f_ping, pingpong_ftn f_pong, int rank0, int rank1, int size, char* sendbuf, 
        char* recvbuf, MPI_Comm comm, MPI_Request* req)
{
    int n_iters = estimate_iters(f_ping, f_pong, rank0, rank1, size, sendbuf, recvbuf, comm, req);
    double time = ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf, recvbuf, n_iters, comm, req);
    return time;
}


void print_ping_pong(pingpong_ftn f_ping, pingpong_ftn f_pong, int max_p, int rank0, int rank1, char* sendbuf, char* recvbuf, 
        MPI_Comm comm, MPI_Request* req)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Print Times
    for (int i = 0; i < max_p; i++)
    {
        int s = pow(2, i);
        if (rank == 0) printf("Size %d: ", s);
        double time = time_ping_pong(f_ping, f_pong, rank0, rank1, s, sendbuf, recvbuf, comm, req);
        MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (rank == 0) printf("%e\n", time);
    }
    if (rank == 0) printf("\n");
}


void standard_ping_pong(int max_p, char* sendbuf, char* recvbuf)
{
    int rank, rank0, rank1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) printf("On-NUMA Ping-Pong:\n");
    rank0 = 0;
    rank1 = PPNUMA/2;
    print_ping_pong(ping, pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, NULL);

    if (rank == 0) printf("On-Socket Ping-Pong:\n");
    rank0 = 0;
    rank1 = PPS/2;
    print_ping_pong(ping, pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, NULL);

    if (rank == 0) printf("On-Node, Off-Socket Ping-Pong:\n");
    rank0 = 0;
    rank1 = PPN/2;
    print_ping_pong(ping, pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, NULL);

    if (rank == 0) printf("Off-Node Ping-Pong:\n");
    rank0 = 0;
    rank1 = PPN;
    print_ping_pong(ping, pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, NULL);
}

void multiproc_ping_pong_step(int max_p, char* sendbuf, char* recvbuf, int max_n, int diff,
    bool cond0, bool cond1)
{
    int rank, rank0, rank1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
        print_ping_pong(ping, pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, NULL);

        // Only test i == size one time, then break
        if (i == max_n)
            break;
    }
}

void multiproc_ping_pong(int max_p, char* sendbuf, char* recvbuf)
{
    int rank, size;
    bool cond0, cond1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // On-NUMA, MultiProc (Standard)
    if (rank == 0) printf("On-NUMA MultiProc Ping-Pong:\n");
    size = PPNUMA/2;
    cond0 = rank < size;
    cond1 = rank < 2*size;
    multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, size, cond0, cond1);

    // On-Socket, MultiProc (Standard)
    if (rank == 0) printf("On-Socket MultiProc Ping-Pong:\n");
    size = PPS/2;
    cond0 = rank < size;
    cond1 = rank < 2*size;
    multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, size, cond0, cond1);
 
    // On-Node, Off-Socket, MultiProc (Standard)
    if (rank == 0) printf("On-Node, Off-Socket MultiProc Ping-Pong:\n");
    size = PPN/2;
    cond0 = rank < size;
    cond1 = rank < 2*size;
    multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, size, cond0, cond1);

    // Off-Node, MultiProc (Standard)
    if (rank == 0) printf("Off-Node MultiProc Ping-Pong:\n");
    size = PPN;
    cond0 = rank < size;
    cond1 = rank < 2*size;
    multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, size, cond0, cond1);

    // On-NUMA, MultiProc (All NUMAs active on 1 Node)
    if (rank == 0) printf("On-NUMA MultiProc Ping-Pong, All NUMAs Active:\n");
    size = PPNUMA/2;
    cond0 = rank < PPN && (rank % PPNUMA) < size;
    cond1 = rank < PPN;
    multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, size, cond0, cond1);

    // On-Socket, MultiProc (All Sockets active on 1 Node) 
    if  (rank == 0) printf("On-Socket MultiProc Ping-Pong, All Sockets Active:\n");
    size = PPS/2;
    cond0 = rank < PPN && (rank % PPS) < size;
    cond1 = rank < PPN;
    multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, size, cond0, cond1);

    // Off-Node, MultiProc, Even NUMA Regions
    if (rank == 0) printf("Off-Node MultiProc Ping-Pong, Even NUMA Regions:\n");
    size = PPNUMA;
    cond0 = rank < PPN;
    cond1 = rank < 2*PPN;
    multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, PPN, cond0, cond1);

    // Off-Node, MultiProc, Even Sockets
    if (rank == 0) printf("Off-Node MultiProc Ping-Pong, Even Sockets:\n");
    size = PPS;
    cond0 = rank < PPN;
    cond1 = rank < 2*PPN;
    multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, PPN, cond0, cond1);
}

void multi_ping_pong(int max_p, char* sendbuf, char* recvbuf, MPI_Request* req)
{   
    int rank, rank0, rank1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) printf("On-NUMA Multi Ping-Pong:\n");
    rank0 = 0;
    rank1 = PPNUMA/2;
    print_ping_pong(multi_ping, multi_pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, req);
    
    if (rank == 0) printf("On-Socket Multi Ping-Pong:\n");
    rank0 = 0;
    rank1 = PPS/2;
    print_ping_pong(multi_ping, multi_pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, req);
    
    if (rank == 0) printf("On-Node, Off-Socket Multi Ping-Pong:\n");
    rank0 = 0;
    rank1 = PPN/2;
    print_ping_pong(multi_ping, multi_pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, req);
    
    if (rank == 0) printf("Off-Node Multi Ping-Pong:\n");
    rank0 = 0;
    rank1 = PPN;
    print_ping_pong(multi_ping, multi_pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, req);
}


void matching_ping_pong(int max_p, char* sendbuf, char* recvbuf, MPI_Request* req)
{   
    int rank, rank0, rank1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
    if (rank == 0) printf("On-NUMA Matching Ping-Pong:\n");
    rank0 = 0;
    rank1 = PPNUMA/2;
    print_ping_pong(matching_ping, matching_pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, req);
    
    if (rank == 0) printf("On-Socket Matching Ping-Pong:\n");
    rank0 = 0;
    rank1 = PPS/2;
    print_ping_pong(matching_ping, matching_pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, req);
    
    if (rank == 0) printf("On-Node, Off-Socket Matching Ping-Pong:\n");
    rank0 = 0;
    rank1 = PPN/2;
    print_ping_pong(matching_ping, matching_pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, req);
    
    if (rank == 0) printf("Off-Node Matching Ping-Pong:\n");
    rank0 = 0;
    rank1 = PPN;
    print_ping_pong(matching_ping, matching_pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, req);
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_p = 25;
    int max_s = pow(2, max_p);
    char* sendbuf = new char[max_s];
    char* recvbuf = new char[max_s];
    MPI_Request* req = new MPI_Request[max_s];

    standard_ping_pong(max_p, sendbuf, recvbuf);

    multiproc_ping_pong(max_p, sendbuf, recvbuf);

    multi_ping_pong(15, sendbuf, recvbuf, req);

    matching_ping_pong(15, sendbuf, recvbuf, req);    

    delete[] sendbuf;
    delete[] recvbuf;
    delete[] req; 

    MPI_Finalize();
    return 0;
}
