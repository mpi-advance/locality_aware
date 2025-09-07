#include <vector>
#include <cstring>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "mpi_advance.h"

template <typename F, typename C>
double time_allreduce(F allreduce_func,
					  const void *sendbuf,
					  void *recvbuf,
					  int count,
					  MPI_Datatype datatype,
					  MPI_Op op,
					  C comm,
					  int n_iters)
{
	MPI_Barrier(MPI_COMM_WORLD);
	double t0 = MPI_Wtime();
	for (int i = 0; i < n_iters; i++)
	{
		allreduce_func(sendbuf, recvbuf, count, datatype, op, comm);
	}

	double tfinal = (MPI_Wtime() - t0) / n_iters;
	MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	return t0;
}

// template <typename F1, typename C>
// double time_multileader_allreduce(F1 allreduce_func,
// 								  const void *sendbuf,
// 								  void *recvbuf,
// 								  int count,
// 								  MPI_Datatype datatype,
// 								  MPI_Op op,
// 								  C comm,
// 								  int n_leaders,
// 								  int n_iters)
// {
// 	MPI_Barrier(MPI_COMM_WORLD);
// 	double t0 = MPI_Wtime();
// 	for (int i = 0; i < n_iters; i++)
// 	{
// 		allreduce_func(sendbuf, recvbuf, count, datatype, op, comm, n_leaders);
// 	}

// 	double tfinal = (MPI_Wtime() - t0) / n_iters;
// 	MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
// 	return t0;
// }

template <typename F, typename C>
double test_allreduce(F allreduce_func,
					  const void *sendbuf,
					  void *recvbuf,
					  int count,
					  MPI_Datatype datatype,
					  MPI_Op op,
					  C comm)
{
	double time;
	int n_iters;

	// Warm-Up
	time_allreduce(allreduce_func, sendbuf, recvbuf, count, datatype, op, comm, 1);

	// Estimate Iterations
	time = time_allreduce(allreduce_func, sendbuf, recvbuf, count, datatype, op, comm, 2);
	MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	n_iters = (1.0 / time) + 1;

	// Time Allreduce
	time = time_allreduce(allreduce_func, sendbuf, recvbuf, count, datatype, op, comm, n_iters);
	MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	return time;
}

// template <typename F1, typename C>
// double test_multileader_allreduce(F1 allreduce_func,
// 								  const void *sendbuf,
// 								  void *recvbuf,
// 								  int count,
// 								  MPI_Datatype datatype,
// 								  MPI_Op op,
// 								  C comm,
// 								  int n_leaders)
// {
// 	double time;
// 	double n_iters;

// 	// Warm-UP
// 	time_multileader_allreduce(allreduce_func, sendbuf, recvbuf, count, datatype, op, comm, 1, n_leaders);

// 	// Estimate Iterations
// 	time = time_multileader_allreduce(allreduce_func, sendbuf, recvbuf, count, datatype, op, comm, 2, n_leaders);
// 	MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
// 	n_iters = (1.0 / time) + 1;

// 	// Time Allreduce
// 	time = time_multileader_allreduce(allreduce_func, sendbuf, recvbuf, count, datatype, op, comm, n_iters, n_leaders);
// 	MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
// 	return time;
// }

template <typename T>
void print_allreduces(int max_p,
					  const T *sendbuf,
					  T *recvbuf,
					  MPI_Datatype datatype,
					  MPI_Op op,
					  MPIX_Comm *comm,
					  T *recvbuf_std)
{
	int rank;
	MPI_Comm_rank(comm->global_comm, &rank);

	int local_rank, ppn;
	MPI_Comm_rank(comm->local_comm, &local_rank);
	MPI_Comm_size(comm->local_comm, &ppn);

	double time;
	using F = int (*)(const void *, void *, int, MPI_Datatype, MPI_Op, _MPIX_Comm);
	std::vector<F> allreduce_funcs = {allreduce_hierarchical, allreduce_node_aware};
	std::vector<const char *> names = {"Hierarchical", "Node Aware"};

	std::vector<F> multileader_allreduce_funcs = {allreduce_multileader, allreduce_locality_aware, allreduce_multileader_locality};
	std::vector<const char *> multileader_names = {"Multileader", "Locality Aware"};

	for (int i = 0; i < max_p; i++)
	{
		int s = pow(2, i);

		if (rank == 0)
		{
			printf("Size %d\n", s);
		}

		// Standard PMP_Allreduce (system MPI)
		PMPI_Allreduce(sendbuf, recvbuf, s, datatype, op, comm->global_comm);
		std::memcpy(recvbuf_std, recvbuf, s * sizeof(T));
		time = test_allreduce(PMPI_Allreduce, sendbuf, recvbuf, s, datatype, op, comm->global_comm);

		if (rank == 0)
		{
			printf("PMPI: %e\n", time);
		}

		// MPI Advance Allreduce functions (not multileader)
		for (int idx = 0; idx < allreduce_funcs.size(); idx++)
		{
			if (rank == 0)
			{
				printf("Testing %s\n", names[idx]);
			}

			allreduce_funcs[idx](sendbuf, recvbuf, s, datatype, op, *comm);
			for (int j = 0; j < s; j++)
			{
				if (fabs(recvbuf_std[j] - recvbuf[j]) > 1e-06)
				{
					printf("Diff Results (%s) %d vs %d\n", names[idx], recvbuf_std[j], recvbuf[j]);
					MPI_Abort(comm->global_comm, -1);
				}
			}

			time = test_allreduce(allreduce_funcs[idx], sendbuf, recvbuf, s, datatype, op, *comm);
			if (rank == 0)
			{
				printf("%s: %e\n", names[idx], time);
			}
		}

		// MPI Advance Multileader Allreduce Functions
		std::vector<int> n_leaders_list = {4, 10, 20};
		for (int ctr = 0; ctr < n_leaders_list.size(); ctr++)
		{
			int n_leaders = n_leaders_list[ctr];
			if (ppn < n_leaders)
			{
				break;
			}

			MPIX_Comm_leader_init(comm, ppn / n_leaders);
			for (int idx = 0; idx < multileader_allreduce_funcs.size(); idx++)
			{
				multileader_allreduce_funcs[idx](sendbuf, recvbuf, s, datatype, op, *comm);
				for (int j = 0; j < s; j++)
				{
					if (fabs(recvbuf_std[j] - recvbuf[j] > 1e-06))
					{
                        printf("DIFF RESULTS (%s) %d vs %d\n", multileader_names[idx], recvbuf_std[j], recvbuf[j]);
						MPI_Abort(comm->global_comm, -1);
					}
				}

				time = test_allreduce(multileader_allreduce_funcs[idx], sendbuf, recvbuf, s, datatype, op, *comm);
				if (rank == 0)
				{
					printf("%s, %d leaders, %e\n", multileader_names[idx], n_leaders, time);
				}
			}
		}
	}
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	int max_p = 11;
	int max_size = pow(2, max_p);

	MPIX_Comm *xcomm;
	MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

	int rank, num_procs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	MPIX_Comm_topo_init(xcomm);

	int local_rank, ppn;
	MPI_Comm_rank(xcomm->local_comm, &local_rank);
	MPI_Comm_size(xcomm->local_comm, &ppn);

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

	print_allreduces(max_p, sendbuf.data(), recvbuf.data(), MPI_FLOAT, MPI_MAX, xcomm, recvbuf_std.data());
}
