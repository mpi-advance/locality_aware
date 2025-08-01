#include <vector>
#include <cstring>

#include <math.h>
#include <mpi.h>
#include "mpi_advance.h"

using A = int(*)(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm);
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

//template <typename F, typename C>
double time_alltoall_subset(A alltoall_func, const void* sendbuf, const int sendcount,
							MPI_Datatype sendtype, void* recvbuf, const int recvcount,
							MPI_Datatype recvtype, MPI_Comm comm, int n_iters)
{
  MPI_Barrier(comm);
  double t0 = MPI_Wtime();
  for (int i = 0; i < n_iters; i++)
	{
	  alltoall_func(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
	}
  double tfinal = (MPI_Wtime() - t0) / n_iters;
  MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, comm);
  return t0;
}

template <typename F, typename C>
int estimate_alltoall_iters(F alltoall_func, const void* sendbuf, int sendcount,
						    MPI_Datatype sendtype, void* recvbuf, int recvcount,
						    MPI_Datatype recvtype, C comm)
{
  double time = time_alltoall(alltoall_func, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, 1);
  MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  int n_iters = 1;
  if (time > 1)
	n_iters = 1;
  else
	{
	  if (time > 1e-01)
		time = time_alltoall(alltoall_func, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, 2);
	  else if (time > 1e-02)
		time = time_alltoall(alltoall_func, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, 10);
	  else
		time = time_alltoall(alltoall_func, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, 100);

	  MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	  n_iters = (1.0 / time) + 1;
	  if (n_iters < 1)
		n_iters = 1;
	}

  return n_iters;
}

double time_gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
				   int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, int n_iters)
{
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();
  for (int i = 0; i < n_iters; i++)
	{
	  MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
	}
  double tfinal = (MPI_Wtime() - t0) / n_iters;
  MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return t0;
}

int estimate_gather_iters(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
						  int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  double time = time_gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, 1);
  MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  int n_iters = 1;
  if (time > 1)
	n_iters = 1;
  else
	{
	  if (time > 1e-01)
		time = time_gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, 2);
	  else if (time > 1e-02)
		time = time_gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, 10);
	  else
		time = time_gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, 100);

	  MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	  n_iters = (1.0 / time) + 1;
	  if (n_iters < 1)
		n_iters = 1;
	}

  return n_iters;
}

double time_scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
					int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, int n_iters)
{
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();
  for (int i = 0; i < n_iters; i++)
	{
	  MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
	}
  double tfinal = (MPI_Wtime() - t0) / n_iters;
  MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return t0;
}

int estimate_scatter_iters(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
							void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
							MPI_Comm comm)
{
  double time = time_scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, 1);
  int n_iters = 1;
  if (time > 1)
	n_iters = 1;
  else
	{
	  if (time > 1e-01)
		time = time_scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, 2);
	  else if (time > 1e-02)
		time = time_scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, 10);
	  else
		time = time_scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, 100);

	  MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	  n_iters = (1.0 / time) + 1;
	  if (n_iters < 1)
		n_iters = 1;
	}

  return n_iters;
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
	n_iters = estimate_alltoall_iters(alltoall_func, sendbuf, sendcount, sendtype,
									  recvbuf, recvcount, recvtype, comm);
	//    time = time_alltoall(alltoall_func, sendbuf, sendcount, sendtype,
	//      recvbuf, recvcount, recvtype, comm, 2);
    //MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    //n_iters = (1.0 / time) + 1;

    // Time Alltoall
    time = time_alltoall(alltoall_func, sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype, comm, n_iters);
    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return time;
}

//template <typename F>
void internal_multileader_timing(A alltoall_func, const void* sendbuf, const int sendcount,
							      MPI_Datatype sendtype, void* recvbuf, const int recvcount,
							      MPI_Datatype recvtype, MPIX_Comm* comm, const char* name,
								  int procsPerLeader)
{
  printf("internal multileader timing\n");
  int rank, num_procs;
  MPI_Comm_rank(comm->global_comm, &rank);
  MPI_Comm_size(comm->global_comm, &num_procs);

  int send_proc, recv_proc;
  int send_pos, recv_pos;
  MPI_Status status;

  char* recv_buffer = (char*) recvbuf;
  char* send_buffer = (char*) sendbuf;

  int send_size, recv_size;
  MPI_Type_size(sendtype, &send_size);
  MPI_Type_size(recvtype, &recv_size);

  if (comm->local_comm == MPI_COMM_NULL)
	MPIX_Comm_topo_init(comm);

  int local_rank, ppn;
  MPI_Comm_rank(comm->local_comm, &local_rank);
  MPI_Comm_size(comm->local_comm, &ppn);

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
	local_recv_buffer = (char*)malloc(sizeof(char));
  }

  int nInternalIters = estimate_gather_iters(send_buffer, sendcount*num_procs, sendtype, local_recv_buffer,
											 sendcount*num_procs, sendtype, 0, comm->local_comm);
  printf("Number of iterations for gather: %d\n", nInternalIters);

  MPI_Barrier(MPI_COMM_WORLD); // should this be local_comm?
  double t0 = MPI_Wtime();
  for (int i = 0; i < nInternalIters; i++)
  {
	MPI_Gather(send_buffer, sendcount*num_procs, sendtype, local_recv_buffer, sendcount*num_procs, sendtype,
			   0, comm->local_comm);
  }

  double time = (MPI_Wtime() - t0) / nInternalIters;
  double tFinalAllgather;
  double tFinalAlltoall;
  MPI_Allreduce(&time, &tFinalAllgather, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if (local_rank == 0)
    {

	  // estimate iterations for alltoall on a subset of ranks
	  time = time_alltoall_subset(alltoall_func, local_send_buffer, ppn * ppn * sendcount, sendtype,
								  local_recv_buffer, ppn * ppn * recvcount, recvtype, comm->group_comm, 1);
	  MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm->group_comm);
	  if (time > 1)
		nInternalIters = 1;
	  else
		{
		  if (time > 1e-01)
			time = time_alltoall_subset(alltoall_func, local_send_buffer, ppn*ppn*sendcount, sendtype,
										local_recv_buffer, ppn*ppn*recvcount, recvtype, comm->group_comm, 2);
		  else if (time > 1e-02)
			time = time_alltoall_subset(alltoall_func, local_send_buffer, ppn*ppn*sendcount, sendtype,
										local_recv_buffer, ppn*ppn*recvcount, recvtype, comm->group_comm, 10);
		  else
			time = time_alltoall_subset(alltoall_func, local_send_buffer, ppn*ppn*sendcount, sendtype,
										local_recv_buffer, ppn*ppn*recvcount, recvtype, comm->group_comm, 100);

		  nInternalIters = (1.0 / time) + 1;
		  if (nInternalIters < 1)
			nInternalIters = 1;
		}

	  printf("nInteralIters for alltoall: %d\n", nInternalIters);
		MPI_Barrier(comm->group_comm);
	    t0 = MPI_Wtime();
		for (int i = 0; i < nInternalIters; i++)
		{
			alltoall_func(local_send_buffer, ppn * ppn * sendcount, sendtype,
						  local_recv_buffer, ppn * ppn * recvcount, recvtype, comm->group_comm);
		}

		time = (MPI_Wtime() - t0) / nInternalIters;
		MPI_Allreduce(&time, &tFinalAlltoall, 1, MPI_DOUBLE, MPI_MAX,  comm->group_comm);
	}

  nInternalIters = estimate_scatter_iters(local_send_buffer, recvcount*num_procs, recvtype, recv_buffer,
										  recvcount*num_procs, recvtype, 0, comm->local_comm);
  printf("nInternalIters for scatter: %d\n", nInternalIters);
	  MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();
	for (int i = 0; i < nInternalIters; i++)
	{
	  MPI_Scatter(local_send_buffer, recvcount * num_procs, recvtype, recv_buffer, recvcount * num_procs, recvtype,
				  0, comm->local_comm);
	}

	time = (MPI_Wtime() - t0) / nInternalIters; 
	double tFinalScatter;
	MPI_Allreduce(&time, &tFinalScatter, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	free(local_send_buffer);
	free(local_recv_buffer);
	
	MPI_Comm_rank(comm->global_comm, &rank);
	if (rank == 0)
	{
	  printf("%s, %d procs per leader, allgather: %e, alltoall: %e, scatter: %e\n", name, procsPerLeader,
			 tFinalAllgather, tFinalAlltoall, tFinalScatter);
	}
}

//template <typename F>
void internal_locality_aware_timing(A alltoallFunc, const void* sendbuf, const int sendcount,
									MPI_Datatype sendtype, void* recvbuf, const int recvcount,
									MPI_Datatype recvtype, MPIX_Comm* comm, const char* name,
									int procsPerLeader)
{
  int rank, num_procs;
  MPI_Comm_rank(comm->global_comm, &rank);
  MPI_Comm_size(comm->global_comm, &num_procs);

  if (comm->local_comm == MPI_COMM_NULL)
	MPIX_Comm_topo_init(comm);
  
  int num_leaders_per_node = 4;
  int procs_per_node;
  MPI_Comm_size(comm->local_comm, &procs_per_node);
  int procs_per_leader = procs_per_node / num_leaders_per_node;
  if (procs_per_node < num_leaders_per_node)
  {
    num_leaders_per_node = procs_per_node;
    procs_per_leader = 1;
  }

  if (comm->leader_comm == MPI_COMM_NULL)
	MPIX_Comm_leader_init(comm, procs_per_leader);
  
  int ppn;
  MPI_Comm_size(comm->leader_comm, &ppn);

  int send_proc, recv_proc;
  int send_pos, recv_pos;
  MPI_Status status;
  
  char* recv_buffer = (char*) recvbuf;
  char* send_buffer = (char*) sendbuf;

  int send_size, recv_size;
  MPI_Type_size(sendtype, &send_size);
  MPI_Type_size(recvtype, &recv_size);

  int n_nodes = num_procs / ppn;

  char* tmpbuf = (char*) malloc(num_procs * sendcount * send_size);

  int nInternalIters = estimate_alltoall_iters(alltoallFunc, sendbuf, ppn*sendcount, sendtype, tmpbuf,
												ppn*recvcount, recvtype, comm->leader_group_comm);
  MPI_Barrier(MPI_COMM_WORLD); 
  double t0 = MPI_Wtime();
  for (int i = 0; i < nInternalIters; i++)
  {
		  alltoallFunc(sendbuf, ppn*sendcount, sendtype, tmpbuf, ppn*recvcount, recvtype, comm->leader_group_comm);
  }

  double time = (MPI_Wtime() - t0) / nInternalIters;
  double tFinalLeaderGroupAlltoall;
  MPI_Allreduce(&time, &tFinalLeaderGroupAlltoall, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


  nInternalIters = estimate_alltoall_iters(alltoallFunc, recvbuf, n_nodes*recvcount, recvtype, tmpbuf, n_nodes*recvcount, recvtype, comm->leader_comm);
  MPI_Barrier(MPI_COMM_WORLD);
  t0 = MPI_Wtime();
  for (int i = 0; i < nInternalIters; i++)
  {
    alltoallFunc(recvbuf, n_nodes * recvcount, recvtype, tmpbuf, n_nodes * recvcount, recvtype, comm->leader_comm);
  }

  time = (MPI_Wtime() - t0) / nInternalIters;
  double tFinalLeaderAlltoall;
  MPI_Allreduce(&time, &tFinalLeaderAlltoall, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  printf("%s, %d procs per leader, leader group comm alltoall: %e, leader comm alltoall: %e\n", name, procsPerLeader,
  		 tFinalLeaderGroupAlltoall, tFinalLeaderAlltoall);  
}

//template <typename F>
void internal_multileader_locality_aware_timing(A alltoallFunc, const void* sendbuf, const int sendcount,
												MPI_Datatype sendtype, void* recvbuf, const int recvcount,
												MPI_Datatype recvtype, MPIX_Comm* comm, const char* name,
												int procsPerLeader)
{
  int rank, num_procs;
  MPI_Comm_rank(comm->global_comm, &rank);
  MPI_Comm_size(comm->global_comm, &num_procs);

  if (comm->local_comm == MPI_COMM_NULL)
	MPIX_Comm_topo_init(comm);
  
  int local_rank, ppn;
  MPI_Comm_rank(comm->local_comm, &local_rank);
  MPI_Comm_size(comm->local_comm, &ppn);

  if (comm->leader_comm == MPI_COMM_NULL)
	{
        int num_leaders_per_node = 4;
        if (ppn < num_leaders_per_node)
            num_leaders_per_node = ppn;
        MPIX_Comm_leader_init(comm, ppn / num_leaders_per_node);
	}

  int procs_per_leader, leader_rank;
  MPI_Comm_rank(comm->leader_comm, &leader_rank);
  MPI_Comm_size(comm->leader_comm, &procs_per_leader);

  int send_proc, recv_proc;
  int send_pos, recv_pos;
  MPI_Status status;

  char* recv_buffer = (char*) recvbuf;
  char* send_buffer = (char*) sendbuf;

  int send_size, recv_size;
  MPI_Type_size(sendtype, &send_size);
  MPI_Type_size(recvtype, &recv_size);

  int n_nodes = num_procs / ppn;
  int n_leaders = num_procs / procs_per_leader;

  int leaders_per_node;
  MPI_Comm_size(comm->leader_local_comm, &leaders_per_node);

  char* local_send_buffer = NULL;
  char* local_recv_buffer = NULL;
  if (leader_rank == 0)
	{
	  local_send_buffer = (char*) malloc(procs_per_leader * num_procs * sendcount * send_size);
	  local_recv_buffer = (char*) malloc(procs_per_leader * num_procs * recvcount * recv_size);
	}
  else
	{
	  local_send_buffer = (char*) malloc(sizeof(char));
	  local_recv_buffer = (char*) malloc(sizeof(char));
	}

  // 1. Local gather
  int nInternalIters = estimate_gather_iters(send_buffer, sendcount*num_procs, sendtype,
										   local_recv_buffer, sendcount*num_procs, sendtype,
										   0, comm->leader_comm);
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();
  for (int i = 0; i < nInternalIters; i++)
	{
	  MPI_Gather(send_buffer, sendcount * num_procs, sendtype, local_recv_buffer, sendcount * num_procs, sendtype,
				 0, comm->leader_comm);
	}
  double time = (MPI_Wtime() - t0) / nInternalIters;
  double tFinalGather;
  MPI_Allreduce(&time, &tFinalGather, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  double tFinalGroupAlltoall;
  double tFinalLeaderLocalAlltoall;
  if (leader_rank == 0)
	{
	  time = time_alltoall_subset(alltoallFunc, local_send_buffer, ppn*procs_per_leader*sendcount, sendtype,
								  local_recv_buffer, ppn*procs_per_leader*recvcount, recvtype, comm->group_comm, 1);
	  MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	  if (time > 1)
		nInternalIters = 1;
	  else
		{
		  if (time > 1e-01)
			time = time_alltoall_subset(alltoallFunc, local_send_buffer, ppn*procs_per_leader*sendcount, sendtype,
										local_recv_buffer, ppn*procs_per_leader*recvcount, recvtype,
										comm->group_comm, 2);
		  else if (time > 1e-02)
			time = time_alltoall_subset(alltoallFunc, local_send_buffer, ppn*procs_per_leader*sendcount, sendtype,
										local_recv_buffer, ppn*procs_per_leader*recvcount, recvtype,
										comm->group_comm, 10);
		  else
			time = time_alltoall_subset(alltoallFunc, local_send_buffer, ppn*procs_per_leader*sendcount, sendtype,
										local_recv_buffer, ppn*procs_per_leader*recvcount, recvtype,
										comm->group_comm, 100);
		  MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm->group_comm);

		  nInternalIters = (1.0 / time) + 1;
		  if (nInternalIters < 1) nInternalIters = 1;
		}
	  
	  MPI_Barrier(comm->group_comm);
	  t0 = MPI_Wtime();
	  for (int i = 0; i < nInternalIters; i++)
		{
		  alltoallFunc(local_send_buffer, ppn * procs_per_leader * sendcount, sendtype,
					   local_recv_buffer, ppn * procs_per_leader * recvcount, recvtype, comm->group_comm);
		}

	  time = (MPI_Wtime() - t0) / nInternalIters;

	  MPI_Allreduce(&time, &tFinalGroupAlltoall, 1, MPI_DOUBLE, MPI_MAX, comm->group_comm);

	  time = time_alltoall_subset(alltoallFunc, local_send_buffer, n_nodes*procs_per_leader*procs_per_leader*sendcount,
								  sendtype, local_recv_buffer, n_nodes*procs_per_leader*procs_per_leader*recvcount,
								  recvtype, comm->leader_local_comm, 1);
	  MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	  if (time > 1)
		nInternalIters = 1;
	  else
		{
		  if (time > 1e-01)
			time = time_alltoall_subset(alltoallFunc, local_send_buffer, n_nodes*procs_per_leader*procs_per_leader*sendcount,
										sendtype, local_recv_buffer, n_nodes*procs_per_leader*procs_per_leader*recvcount,
										recvtype, comm->leader_local_comm, 2);
		  else if (time > 1e-02)
			time = time_alltoall_subset(alltoallFunc, local_send_buffer, n_nodes*procs_per_leader*procs_per_leader*sendcount,
										sendtype, local_recv_buffer, n_nodes*procs_per_leader*procs_per_leader*recvcount,
										recvtype, comm->leader_local_comm, 10);
		  else
			time = time_alltoall_subset(alltoallFunc, local_send_buffer, n_nodes*procs_per_leader*procs_per_leader*recvcount,
										sendtype, local_recv_buffer, n_nodes*procs_per_leader*procs_per_leader*recvcount,
										recvtype, comm->leader_local_comm, 100);
		  nInternalIters = (1.0 / time) + 1;
		  if (nInternalIters < 1) nInternalIters = 1;
		}
	  
	  MPI_Barrier(comm->leader_local_comm);
	  t0 = MPI_Wtime();
	  for (int i = 0; i < nInternalIters; i++)
		{
		  alltoallFunc(local_send_buffer, n_nodes * procs_per_leader * procs_per_leader * sendcount, sendtype,
					   local_recv_buffer, n_nodes * procs_per_leader * procs_per_leader * recvcount, recvtype,
					   comm->leader_local_comm);
		}

	  time = (MPI_Wtime() - t0) / nInternalIters;
	  MPI_Allreduce(&time, &tFinalLeaderLocalAlltoall, 1, MPI_DOUBLE, MPI_MAX, comm->leader_local_comm);
	}

  nInternalIters = estimate_scatter_iters(local_send_buffer, recvcount*num_procs, recvtype, recv_buffer,
										  recvcount*num_procs, recvtype, 0, comm->leader_comm);
  MPI_Barrier(MPI_COMM_WORLD);
  t0 = MPI_Wtime();
  for (int i = 0; i < nInternalIters; i++)
	{
	  MPI_Scatter(local_send_buffer, recvcount * num_procs, recvtype, recv_buffer, recvcount * num_procs, recvtype,
				  0, comm->leader_comm);
	}
  time = (MPI_Wtime() - t0) / nInternalIters;
  double tFinalScatter;
  MPI_Allreduce(&time, &tFinalScatter, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if (rank == 0)
	{
	  printf("%s, %d procs per leader, all gather: %e, group alltoall: %e, leader local alltoall: %e, scatter: %e\n",
			 name, procsPerLeader, tFinalGather, tFinalGroupAlltoall, tFinalLeaderLocalAlltoall, tFinalScatter);
	}
}

//template <typename F>
void no_op_internal_timing(A alltoallFunc, const void* sendbuff, const int sendcount, MPI_Datatype sendtype,
						  void *recvbuf, const int recvcount, MPI_Datatype recvtype, MPIX_Comm* comm,
						  const char* name, int procsPerLeader)
{
  // no op
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
	using I = void (*)(A, const void*, const int, MPI_Datatype, void*, const int, MPI_Datatype, _MPIX_Comm*, const char*, int);
    std::vector<F> alltoall_funcs = {alltoall_pairwise, alltoall_nonblocking, alltoall_hierarchical, alltoall_node_aware, alltoall_hierarchical_nb, alltoall_node_aware_nb};
    std::vector<const char*> names = {"Pairwise", "NonBlocking", "Pairwise Hierarchical", "Pairwise Node Aware", "Nonblocking Hierarchical", "Nonblocking Node Aware"};
	std::vector<I> timingFuncs = {no_op_internal_timing, no_op_internal_timing, internal_multileader_timing, internal_locality_aware_timing, internal_multileader_timing, internal_locality_aware_timing};
	std::vector<A> internalAlltoallFuncs = {pairwise_helper, nonblocking_helper, pairwise_helper, pairwise_helper, nonblocking_helper, nonblocking_helper};

    std::vector<F> multileader_funcs = { alltoall_multileader, alltoall_locality_aware, alltoall_multileader_locality, alltoall_multileader_nb, alltoall_locality_aware_nb, alltoall_multileader_locality_nb};
    std::vector<const char*> multileader_names = {"Pairwise Multileader", "Pairwise Locality Aware", "Pairwise Multileader Locality", "Nonblocking Multileader", "Nonblocking Locality Aware", "Nonblocking Multileader Locality"};
	std::vector<I> multileaderTimingFuncs = {internal_multileader_timing, internal_locality_aware_timing, internal_multileader_locality_aware_timing, internal_multileader_timing, internal_locality_aware_timing, internal_multileader_locality_aware_timing};
	std::vector<A> multileaderInternalAlltoallFuncs = {pairwise_helper, pairwise_helper, pairwise_helper, nonblocking_helper, nonblocking_helper, nonblocking_helper};
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
			timingFuncs[idx](internalAlltoallFuncs[idx], sendbuf, s, sendtype, recvbuf, s, recvtype, comm, names[idx], 1);
        }

        // MPI Advance Multileader Alltoall Functions
		//				std::vector<int> procs_per_leader_list = {4, 8, 16};
        //for (int ctr = 0; ctr < procs_per_leader_list.size(); ctr++)
        //{
		//int n_procs = procs_per_leader_list[ctr];
		//if (ppn < n_procs)
		//    break;
		//MPIX_Comm_leader_init(comm, n_procs);
		//
		//for (int idx = 0; idx < multileader_funcs.size(); idx++)
		//{   
		//    multileader_funcs[idx](sendbuf, s, sendtype, recvbuf, s, recvtype, comm);
		//    for (int j = 0; j < s; j++) 
		//        if (fabs(recvbuf_std[j] - recvbuf[j]) > 1e-06)
		//        {   
		//            printf("DIFF RESULTS %d vs %d\n", recvbuf_std[j], recvbuf[j]);
		//            MPI_Abort(comm->global_comm, -1);
		//        }
		//    time = test_alltoall(multileader_funcs[idx], sendbuf, s, sendtype,
		//            recvbuf, s, recvtype, comm);
		//    if (rank == 0) printf("%s, %d procs per leader: %e\n", multileader_names[idx], n_procs, time);
		//		multileaderTimingFuncs[idx](multileaderInternalAlltoallFuncs[idx], sendbuf, s, sendtype, recvbuf, s, recvtype, comm, multileader_names[idx], n_procs);
		//}

		//  MPIX_Comm_leader_free(comm);
		//        }
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
