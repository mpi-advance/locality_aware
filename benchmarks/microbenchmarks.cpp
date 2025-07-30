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
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, comm);
    return t0;
}

int estimate_iters(pingpong_ftn f_ping, pingpong_ftn f_pong, int rank0, int rank1, int size, char* sendbuf, 
        char* recvbuf, MPI_Comm comm, MPI_Request* req)
{   
    double time;
    
    // Time a single iteration
    time = ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf, recvbuf, 1, comm, req);
    
    // If single iteration takes longer than 1 second, we only want 1 iteration
    if (time > 1.0)
        return 1.0;
    
    // If we will need less than 10 iterations, use 2 iterations to estimate iteration count
    if (time > 1e-01)
    {   
        time = ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf, recvbuf, 2, comm, req);
    }
    
    // Otherwise, If we want less than 100 iterations, use 10 iterations to estimate iteration count
    else if (time > 1e-02)
    {   
        time = ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf, recvbuf, 10, comm, req);
    }
    
    // Otherwise, use 100 iterations to estimate iteration count
    else
    {   
        time = ping_pong(f_ping, f_pong, rank0, rank1, size, sendbuf, recvbuf, 100, comm, req);
    }
    
    int n_iters = (1.0 / time) + 1;
    if (n_iters < 1) n_iters = 1;
    
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
        if (rank == 0) printf("%e\n", time);
    }
    if (rank == 0) printf("\n");
}


void standard_ping_pong(int max_p, char* sendbuf, char* recvbuf)
{
    int rank, rank0, rank1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (PPNUMA <= 1)
    {
        if (rank == 0) printf("Not measuring intra-NUMA, because PPNUMA is set to %d\n", PPNUMA);
    }
    else if (PPNUMA == PPN)
    {
        if (rank == 0) printf("Not measuring intra-NUMA, because only 1 NUMA per node\n");
    }
    else
    {
        if (rank == 0) printf("On-NUMA Ping-Pong:\n");
        rank0 = 0;
        rank1 = PPNUMA/2;
        print_ping_pong(ping, pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, NULL);
    }


    if (PPS <= 1)
    {
        if (rank == 0) printf("Not measuring intra-socket, because PPS is set to %d\n", PPS);
    }
    else if (PPS == PPN)
    {
        if (rank == 0) printf("Not measuring intra-socket, because only 1 socket per node\n");
    }
    else if (PPNUMA == PPS)
    {
        if (rank == 0) printf("Not measuring intra-socket because same as intra-NUMA\n");
    }
    else
    {
        if (rank == 0) printf("On-Socket Ping-Pong:\n");
        rank0 = 0;
        rank1 = PPS/2;
        print_ping_pong(ping, pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, NULL);
    }

    if (PPN <= 1)
    {
        if (rank == 0) printf("Not measuring intra-node, because PPN is set to %d\n", PPN);
    }
    else
    {
        if (rank == 0) printf("On-Node, Off-Socket Ping-Pong:\n");
        rank0 = 0;
        rank1 = PPN/2;
        print_ping_pong(ping, pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, NULL);
    }

    if (NODES < 2)
    {
        printf("Not measuring inter-node because NODES is set to %d\n", NODES);
    }
    else
    {
        if (rank == 0) printf("Off-Node Ping-Pong:\n");
        rank0 = 0;
        rank1 = PPN;
        print_ping_pong(ping, pong, max_p, rank0, rank1, sendbuf, recvbuf, MPI_COMM_WORLD, NULL);
    }
}

void multiproc_ping_pong_step(int max_p, char* sendbuf, char* recvbuf, int max_n, int diff, int div,
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

if (rank == 0) printf("Rank %d sending to %d\n", rank0, rank1);

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
    
    if (PPNUMA <= 1)
    {
        if (rank == 0) printf("Not measuring multi-proc intra-NUMA, because PPNUMA is set to %d\n", PPNUMA);
    }
    else if (PPNUMA == PPN)
    {
        if (rank == 0) printf("Not measuring multi-proc intra-NUMA, because only 1 NUMA per node\n");
    }
    else
    {
        if (rank == 0) printf("On-NUMA MultiProc Ping-Pong:\n");
        size = PPNUMA/2;
        cond0 = rank < size;
        cond1 = rank < 2*size;
        multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, size, 1, cond0, cond1);
    }

    // On-Socket, MultiProc (Standard)
    if (PPS <= 1)
    {
        if (rank == 0) printf("Not measuring multi-proc intra-socket, because PPS is set to %d\n", PPS);
    }
    else if (PPS == PPN)
    {
        if (rank == 0) printf("Not measuring multi-proc intra-socket, because only 1 socket per node\n");
    }
    else if (PPNUMA == PPS)
    {
        if (rank == 0) printf("Not measuring multi-proc intra-socket because same as intra-NUMA\n");
    }
    else
    {
        if (rank == 0) printf("On-Socket MultiProc Ping-Pong:\n");
        size = PPS/2;
        cond0 = rank < size;
        cond1 = rank < 2*size;
        multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, size, 1, cond0, cond1);
     }


    // On-Node, Off-Socket, MultiProc (Standard)
    if (PPN <= 1)
    {
        if (rank == 0) printf("Not measuring multi-proc intra-node, because PPN is set to %d\n", PPN);
    }
    else
    {
        if (rank == 0) printf("On-Node, Off-Socket MultiProc Ping-Pong:\n");
        size = PPN/2;
        cond0 = rank < size;
        cond1 = rank < 2*size;
        multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, size, 1, cond0, cond1);
    }


    // Off-Node, MultiProc (Standard)
    if (NODES < 2)
    {
        printf("Not measuring inter-node because NODES is set to %d\n", NODES);
    }
    else
    {
        if (rank == 0) printf("Off-Node MultiProc Ping-Pong:\n");
        size = PPN;
        cond0 = rank < size;
        cond1 = rank < 2*size;
        multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, size, 1, cond0, cond1);
    }

    // On-NUMA, MultiProc (All NUMAs active on 1 Node)
    if (PPNUMA <= 1)
    {
        if (rank == 0) printf("Not measuring intra-numa multiproc with all numas active, because PPNUMA is set to %d\n", PPNUMA);
    }
    else if (PPNUMA == PPN)
    {
        if (rank == 0) printf("Not measuring intra-numa multiproc with all numas active, because only 1 NUMA per node\n");
    }
    else
    {
        if (rank == 0) printf("On-NUMA MultiProc Ping-Pong, All NUMAs Active:\n");
        size = PPNUMA/2;
        cond0 = rank < PPN && (rank % PPNUMA) < size;
        cond1 = rank < PPN;
        multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, size, 1, cond0, cond1);
    }

    // On-Socket, MultiProc (All Sockets active on 1 Node) 
    if (PPS <= 1)
    {
        if (rank == 0) printf("Not measuring multi-proc intra-socket with all numas active, because PPS is set to %d\n", PPS);
    }
    else if (PPS == PPN)
    {
        if (rank == 0) printf("Not measuring multi-proc intra-socket with all numas active, because only 1 socket per node\n");
    }
    else if (PPNUMA == PPS)
    {
        if (rank == 0) printf("Not measuring multi-proc intra-socket with all numas active, because same as intra-NUMA\n");
    }
    else
    {
        if  (rank == 0) printf("On-Socket MultiProc Ping-Pong, All Sockets Active:\n");
        size = PPS/2;
        cond0 = rank < PPN && (rank % PPS) < size;
        cond1 = rank < PPN;
        multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, size, 1, cond0, cond1);
    }

    // Off-Node, MultiProc, Even NUMA Regions
    if (NODES < 2)
    {
        if (rank == 0) printf("Not measuring off-node multiproc with even NUMAs, because NODES is set to %d\n", NODES);
    }
    else if (PPN == PPNUMA)
    {
        if (rank == 0) printf("Not measuring off-node multiproc with even NUMAs, because only 1 NUMA per node\n");
    }
    else
    {
        if (rank == 0) printf("Off-Node MultiProc Ping-Pong, Even NUMA Regions:\n");
        size = PPNUMA;
        cond0 = rank < PPN;
        cond1 = rank < 2*PPN;
        multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, PPN, 1, cond0, cond1);
    }

    // Off-Node, MultiProc, Even Sockets
    if (NODES < 2)
    {
        if (rank == 0) printf("Not measuring off-node multiproc with even sockets, because NODES is set to %d\n", NODES);
    }
    else if (PPN == PPS)
    {
        if (rank == 0) printf("Not measuring off-node multiproc with even sockets, because only 1 socket per node\n");
    }
    else if (PPS == PPNUMA)
    {
        if (rank == 0) printf("Not measuring off-node multiproc with even sockets, because same as even NUMAs\n");
    }
    else
    {
        if (rank == 0) printf("Off-Node MultiProc Ping-Pong, Even Sockets:\n");
        size = PPS;
        cond0 = rank < PPN;
        cond1 = rank < 2*PPN;
        multiproc_ping_pong_step(max_p, sendbuf, recvbuf, size, PPN, 1, cond0, cond1);
    }
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

    if (rank == 0) 
    {
        printf("Inter-CPU Microbenchmarks will be run with the following:\n");
        printf("# Processes per NUMA: %d\n", PPNUMA);
        printf("# Processes per socket: %d\n", PPS);
        printf("# Processes per node: %d\n", PPN);
        printf("# of nodes: %d\n", NODES); 
        printf("If these are incorrect, edit the defines at the top of benchmarks/microbenchmarks.cpp\n");

        if (PPNUMA <= 1)
        {
            printf("intra-NUMA tests will not be run because PPNUMA is set to %d.\n", PPNUMA);
        }
        else if (PPNUMA == PPN)
        {
            printf("intra-NUMA tests will not be run because there is only one NUMA per node.  The timings will be gathered in the intra-node tests.\n");
        }

        if (PPS <= 1)
        {
            printf("intra-socket tests will not be run because PPS is set to %d.\n", PPS);
        }
        else if (PPS == PPN)
        {
            printf("intra-socket tests will not be run because there is only one socket per node.  The timings will be gathered in inter-node tests.\n");
        }
        else if (PPS == PPNUMA)
        {
            printf("intra-socket tests will not be run because there is only one NUMA per socket.  The timings will be gathered in inter-NUMA tests.\n");
        }

        if (PPN <= 1)
        {
            printf("intra-node tests will not be run because PPN is set to %d\n", PPN);
        }

        if (NODES < 2)
        {
            printf("inter-node tests will not be run because NODES is set to %d\n", NODES);
        }

        printf("\n\n");
    }

    if (rank == 0) printf("Running standard ping-pongs\n\n");
    standard_ping_pong(max_p, sendbuf, recvbuf);

    if (rank == 0) printf("Running multi-proc ping pongs\n\n");
    multiproc_ping_pong(max_p, sendbuf, recvbuf);

#if MATCHING
    if (rank == 0) printf("Running standard many message tests (with ordered tags)\n"); 
    multi_ping_pong(15, sendbuf, recvbuf, req);

    if (rank == 0) printf("Running unordered many messages tests to stress matching costs\n");
    matching_ping_pong(15, sendbuf, recvbuf, req);    
#else
   if (rank == 0) printf("Not running any matching (queue search) benchmarks.  To run these, edit the #define MATCHING to a nonzero number\n");
#endif

    delete[] sendbuf;
    delete[] recvbuf;
    delete[] req; 

    MPI_Finalize();
    return 0;
}
