#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
class PingPong
{
    public:

    PingPong(int _proc, MPI_Request* _request, int _even_odd)
    {
        proc = _proc;
        request = _request;
        tag = 0;
        even_odd = _even_odd;
        time = 0;
        sendbuf = 1;
        recvbuf = 0;
    }

    void start(int _n_iter)
    {
        n_iter = _n_iter;
        time = MPI_Wtime();
        step();
    }

    void ping()
    {
        MPI_Isend(&sendbuf, 1, MPI_FLOAT, proc, tag++, MPI_COMM_WORLD, request);
    }

    void pong()
    {
        MPI_Irecv(&recvbuf, 1, MPI_FLOAT, proc, tag++, MPI_COMM_WORLD, request);
    }

    int step()
    {
        if (tag / 2 == n_iter)
        {
            time = MPI_Wtime() - time;
            return 0;
        }

        if (tag % 2 == even_odd)
        {
            ping();
        }
        else
        {
            pong();
        }

        return 1;
    }


    int proc;
    int even_odd;
    float sendbuf;
    float recvbuf;
    MPI_Request* request;
    int tag;
    double time;
    int n_iter;
};

void dual_ping_pongs(PingPong** ping_pong, MPI_Request* req, int n_iter)
{
    // Start both ping pongs
    ping_pong[0]->start(n_iter);
    ping_pong[1]->start(n_iter);

    // Progress the ping pongs until n_iter iterations complete
    int active = 1;
    int idx;
    while (active)
    {
        // Wait for the current step of either ping pong to complete
        MPI_Waitany(2, req, &idx, MPI_STATUS_IGNORE);

        // Progress that ping pong
        active = ping_pong[idx]->step();
    }

    // Once a ping pong complete, progress only the other ping pong
    idx = (idx + 1) % 2;
    active = 1;
    while (active)
    {
        MPI_Wait(&(req[idx]), MPI_STATUS_IGNORE);
        active = ping_pong[idx]->step();
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int send_proc, recv_proc;

    PingPong* ping_pong[2];
    MPI_Request req[2];

    double times[num_procs];
    times[rank] = 0.0;
    for (int i = 1; i < num_procs; i++)
    {
        send_proc = (rank + i) % num_procs;
        recv_proc = (rank - i + num_procs) % num_procs;

        // Initialize Ping Pongs
        // I time ping_pong[0] and only participate in ping_pong[1]
        ping_pong[0] = new PingPong(send_proc, &(req[0]), 0);
        ping_pong[1] = new PingPong(recv_proc, &(req[1]), 1);

        // Warm-Up
        dual_ping_pongs(ping_pong, req, 1);

        // Time 100 Iterations
        dual_ping_pongs(ping_pong, req, 100000);
	//        printf("Ping Pong [%d to %d]: %e\n", rank, send_proc, ping_pong[0]->time);
        times[send_proc] = ping_pong[0]->time;
        delete ping_pong[0];
        delete ping_pong[1];
    }

    double* adjacencyMatrix = (double*) malloc(num_procs * num_procs * sizeof(double));
    MPI_Gather(times, num_procs, MPI_DOUBLE, adjacencyMatrix, num_procs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0)
    {
        printf("Adjacency Matrix\n\n");
	for (int i = 0; i < num_procs; i++)
	  {
	    for (int j = 0; j < num_procs; j++)
	      {
		printf("%.10lf\t", adjacencyMatrix[i * num_procs + j]);
	      }
	    printf("\n");
	  }
    }

    MPI_Finalize();
    
    return 0;
}
