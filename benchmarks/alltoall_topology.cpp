#include <vector>
#include <cstring>

#include <math.h>
#include <mpi.h>
#include "mpi_advance.h"

template <typename F, typename C>
double time_alltoall(F alltoall_func, const void* sendbuf, const int sendcount,
        MPI_Datatype sendtype, void* recvbuf, const int recvcount, MPI_Datatype recvtype,
        C comm, int n_iters)
{
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        alltoall_func(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    }
    double tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return t0;
}

template <typename F, typename C>
double test_alltoall(F alltoall_func, const void* sendbuf, const int sendcount,
        MPI_Datatype sendtype, void* recvbuf, const int recvcount, MPI_Datatype recvtype,
        C comm)
{
    double time;
    int n_iters;

    // Warm-Up
    time_alltoall(alltoall_func, sendbuf, sendcount, sendtype, 
            recvbuf, recvcount, recvtype, comm, 1);

    // Estimate Iterations
    time = time_alltoall(alltoall_func, sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm, 2);
    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    n_iters = (1.0 / time) + 1;

    // Time Alltoall
    time = time_alltoall(alltoall_func, sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm, n_iters);
    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return time;
}

template <typename F>
double internal_multileader_timing(F alltoall_func, const void* sendbuf, const int sendcount,
							       MPI_Datatype sendtype, void* recvbuf, const int recvcount,
							       MPI_Datatype recvtype, MPIX_Comm* comm, const char* name,
								   int numLeaders)
{
  double time;

  int rank, num_procs;
  MPI_Comm_rank(comm->global_comm, &rank);
  MPI_Comm_size(comm->global_comm, &num_procs);

  int local_rank, ppn;
  MPI_Comm_rank(comm->local_comm, &local_rank);
  MPI_Comm_size(comm->local_comm, &ppn);

  int send_size, recv_size;
  MPI_Type_size(sendtype, &send_size);
  MPI_Type_size(recvtype, &recv_size);
  
  int n_nodes = num_procs / ppn;

  char* local_send_buffer = NULL;
  char* local_recv_buffer = NULL;

  if (local_rank == 0)
  {
	local_send_buffer = (char*)malloc(ppn*num_procs*sendcount*send_size);
	local_recv_buffer = (char*)malloc(ppn*num_procs*recvcount*recv_size);
  }
  else
  {
	local_send_buffer = (char*)malloc(sizeof(char));
	local_recv_buffer = (char*)malloc(sidzeof(char));
  }

  char* recv_buffer = (char*)recvbuf;
  char* send_buffer = (char*)sendbuf;
  
  MPI_Barrier(MPI_COMM_WORLD); // should this be local_comm?
  double t0 = MPI_Wtime();
  for (int i = 0; i < 1000; i++)
  {
	MPI_Gather(send_buffer, sendcount*num_procs, sendtype, local_recv_buffer, sendcount*num_procs, sendtype,
			   0, comm->local_comm);
  }

  double time = (MPI_Wtime() - t0) / 1000;
  double tFinalAllgather;
  MPI_Allreduce(&time, &tFinalAllgather, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  // 2. Re-pack for sends
    // Assumes SMP ordering 
    // TODO: allow for other orderings
    int ctr;

    if (local_rank == 0)
    {
        ctr = 0;
        for (int dest_node = 0; dest_node < n_nodes; dest_node++)
        {
            int dest_node_start = dest_node * ppn * sendcount * send_size;
            for (int origin_proc = 0; origin_proc < ppn; origin_proc++)
            {
                int origin_proc_start = origin_proc * num_procs * sendcount * send_size;
                memcpy(&(local_send_buffer[ctr]), &(local_recv_buffer[origin_proc_start + dest_node_start]),
                        ppn * sendcount * send_size);
                ctr += ppn * sendcount * send_size;
            }
        }

		MPI_Barrier(comm->group_comm);
	    t0 = MPI_Wtime();
		for (int i = 0; i < 1000; i++)
		{
			alltoall_func(local_send_buffer, ppn * ppn * sendcount, sendtype,
						  local_recv_buffer, ppn * ppn * recvcount, recvtype, comm->group_comm);
		}

		time = (MPI_Wtime() - t0) / 1000;
		double tFinalAlltoall;
		MPI_Allreduce(&time, &tFinalAlltoall, 1, MPI_DOUBLE, MPI_MAX,  comm->group_comm);

		ctr = 0;
		for (int dest_proc = 0; dest_proc < ppn; dest_proc++)
        {
            int dest_proc_start = dest_proc * recvcount * recv_size;
            for (int orig_proc = 0; orig_proc < num_procs; orig_proc++)
            {
                int orig_proc_start = orig_proc * ppn * recvcount * recv_size;
                memcpy(&(local_send_buffer[ctr]), &(local_recv_buffer[orig_proc_start + dest_proc_start]),
                        recvcount * recv_size);
                ctr += recvcount * recv_size;

            }
        }
	}

	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	for (int i = 0; i < 1000; i++)
	{
	  MPI_Scatter(local_send_buffer, recvcount * num_procs, recvtype, recv_buffer, recvcount * num_procs, recvtype,
				  0, comm->local_comm);
	}
	time = (MPI_Wtime - t0) / 1000; 
	double tFinalScatter;
	MPIAllreduce(&time, &tFinalScatter, 1, MPI_DOUBLE, MPI_MAX, comm->comm_world);


	free(local_send_buffer);
	free(local_recv_buffer);
	
	MPI_Comm_rank(comm->global_comm, &rank);
	if (rank == 0)
	{
	  printf("%s, %d procs per leader, allgather: %e, alltoall: %e, scatter: %e\n", name, procsPerLeader,
			 tFinalAllgather, tFinalAlltoall, tFinalScatter);
	}
}

template <typename T>
void print_alltoalls(int max_p, const T* sendbuf,  
        MPI_Datatype sendtype, T* recvbuf, MPI_Datatype recvtype,
        MPIX_Comm* comm, T* recvbuf_std)
{
    int rank;
    MPI_Comm_rank(comm->global_comm, &rank);

    int local_rank, ppn;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &ppn);

    double time;

    using F = int (*)(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, _MPIX_Comm*);
    std::vector<F> alltoall_funcs = {alltoall_pairwise, alltoall_nonblocking, alltoall_hierarchical, alltoall_node_aware, alltoall_hierarchical_nb, alltoall_node_aware_nb};
    std::vector<const char*> names = {"Pairwise", "NonBlocking", "Pairwise Hierarchical", "Pairwise Node Aware", "Nonblocking Hierarchical", "Nonblocking Node Aware"};

    std::vector<F> multileader_funcs = { alltoall_multileader, alltoall_locality_aware, alltoall_multileader_locality, alltoall_multileader_nb, alltoall_locality_aware_nb, alltoall_multileader_locality_nb};
    std::vector<const char*> multileader_names = {"Pairwise Multileader", "Pairwise Locality Aware", "Pairwise Multileader Locality", "Nonblocking Multileader", "Nonblocking Locality Aware", "Nonblocking Multileader Locality"};

    for (int i = 0; i < max_p; i++)
    {
        int s = pow(2, i);

        if (rank == 0) printf("Size %d\n", s);
        
        // Standard PMPI Alltoall (system MPI)
        PMPI_Alltoall(sendbuf, s, sendtype, recvbuf, s, recvtype, comm->global_comm);
        std::memcpy(recvbuf_std, recvbuf, s*sizeof(T));
        time = test_alltoall(PMPI_Alltoall, sendbuf, s, sendtype,
                recvbuf, s, recvtype, comm->global_comm);
        if (rank == 0) printf("PMPI: %e\n", time);

        // MPI Advance Alltoall Functions (not multileader)
        for (int idx = 0; idx < alltoall_funcs.size(); idx++)
        {
            alltoall_funcs[idx](sendbuf, s, sendtype, recvbuf, s, recvtype, comm); 
            for (int j = 0; j < s; j++)
                if (fabs(recvbuf_std[j] - recvbuf[j]) > 1e-06)
                {
                    printf("DIFF RESULTS %d vs %d\n", recvbuf_std[j], recvbuf[j]);
                    MPI_Abort(comm->global_comm, -1);
                }
            time = test_alltoall(alltoall_funcs[idx], sendbuf, s, sendtype,
                    recvbuf, s, recvtype, comm);
            if (rank == 0) printf("%s: %e\n", names[idx], time);
        }

        // MPI Advance Multileader Alltoall Functions
		std::vector<int> procs_per_leader_list = {4, 8, 16};
        for (int ctr = 0; ctr < procs_per_leader_list.size(); ctr++)
        {
		    int n_procs = procs_per_leader_list[ctr];
		    if (ppn < n_procs)
                break;
            MPIX_Comm_leader_init(comm, n_procs);

            for (int idx = 0; idx < multileader_funcs.size(); idx++)
            {   
                multileader_funcs[idx](sendbuf, s, sendtype, recvbuf, s, recvtype, comm);
                for (int j = 0; j < s; j++) 
                    if (fabs(recvbuf_std[j] - recvbuf[j]) > 1e-06)
                    {   
                        printf("DIFF RESULTS %d vs %d\n", recvbuf_std[j], recvbuf[j]);
                        MPI_Abort(comm->global_comm, -1);
                    }
                time = test_alltoall(multileader_funcs[idx], sendbuf, s, sendtype,
                        recvbuf, s, recvtype, comm);
                if (rank == 0) printf("%s, %d procs per leader: %e\n", multileader_names[idx], n_procs, time);
            }

            MPIX_Comm_leader_free(comm);
        }
    }

}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int max_p = 15;
    int max_size = pow(2, max_p);

    MPIX_Comm* xcomm;
    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPIX_Comm_topo_init(xcomm);

    int local_rank, ppn;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_size(xcomm->local_comm, &ppn);

    // To test a different number of leaders, change here: 
    // TODO : currently need num_leaders_per_node to evenly divide ppn
    std::vector<int> sendbuf(max_size * num_procs);
    std::vector<int> recvbuf(max_size * num_procs);
    std::vector<int> recvbuf_std(max_size * num_procs);

    for (int j = 0; j < num_procs; j++)
    {
        for (int k = 0; k < max_size; k++)
        {
            sendbuf[j * max_size + k] = rank * 10000 + j * 100 + k;
        }
    }

    print_alltoalls(max_p, sendbuf.data(), MPI_FLOAT, recvbuf.data(), MPI_FLOAT, 
            xcomm, recvbuf_std.data()); 

}
