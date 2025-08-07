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
    std::vector<F> alltoall_funcs = {alltoall_hierarchical, alltoall_node_aware, alltoall_hierarchical_nb, alltoall_node_aware_nb};
    std::vector<const char*> names = {"Pairwise Hierarchical", "Pairwise Node Aware", "Nonblocking Hierarchical", "Nonblocking Node Aware"};

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
		  if (rank == 0)
			printf("Testing %s\n", names[idx]);
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
		if (rank == 0)
		  printf("Testing multileader functions\n");
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

    int max_p = 2;
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
