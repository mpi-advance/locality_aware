#include <vector>
#include <cstring>

#include <math.h>
#include <mpi.h>
#include "mpi_advance.h"

template <typename F, typename... Args>
double time_collective(F func, int n_iters, bool participant, Args&&... args)
{
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();
  for (int i = 0; i < n_iters; i++)
  {
    if (participant)
        func(std::forward<Args>(args)...);
  }
  double tfinal = (MPI_Wtime() - t0) / n_iters;
  MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return t0;
}

template <typename F, typename... Args>
int estimate_collective_iters(F collective_func, bool participant, Args&&... args)
{
  double time = time_collective(collective_func, 1, participant, std::forward<Args>(args)...);
  MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  int n_iters = 1;
  if (time > 1)
    n_iters = 1;
  else
  {
    if (time > 1e-01)
      n_iters = 2;
    else if (time > 1e-02)
      n_iters = 10;
    else
      n_iters = 100;
    time = time_collective(collective_func, n_iters, participant, std::forward<Args>(args)...);
    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    n_iters = (1.0 / time) + 1;
    if (n_iters < 1)
      n_iters = 1;
  }

  return n_iters;
}

template <typename F, typename... Args>
double test_collective(F func, bool participant, Args&&... args)
{
  double time;
  int n_iters;

  // Warm-Up
  time_collective(func, 1, participant, std::forward<Args>(args)...);

  // Estimate Iterations
  n_iters = estimate_collective_iters(func, participant, std::forward<Args>(args)...);

  // Time Alltoall
  time = time_collective(func, n_iters, participant, std::forward<Args>(args)...);
  MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return time;
}

template <typename T>
void print_alltoalls(int max_p, const T *sendbuf,
                     MPI_Datatype sendtype, T *recvbuf, MPI_Datatype recvtype,
                     MPIX_Comm *comm, T *recvbuf_std)
{
  int rank, num_procs;
  MPI_Comm_rank(comm->global_comm, &rank);
  MPI_Comm_size(comm->global_comm, &num_procs);

  int local_rank, ppn;
  MPI_Comm_rank(comm->local_comm, &local_rank);
  MPI_Comm_size(comm->local_comm, &ppn);

  int my_node, n_nodes;
  MPI_Comm_rank(comm->group_comm, &my_node);
  MPI_Comm_size(comm->group_comm, &n_nodes);

  int max_s = 1 << max_p;

  double time;
  int tag;

  T* local_sendbuf = NULL;
  T* local_recvbuf = NULL;

  for (int i = 0; i < max_p; i++)
  {
    int s = 1 << i;

    if (local_rank == 0)
    {
      local_sendbuf = (T*)malloc(s * num_procs * ppn * sizeof(T));
      local_recvbuf = (T*)malloc(s * num_procs * ppn * sizeof(T));
    }

    if (rank == 0)
      printf("Size %d\n", s);

    // Standard PMPI Alltoall (system MPI)
    time = test_collective(PMPI_Alltoall, true, sendbuf, s, sendtype,
                         recvbuf, s, recvtype, comm->global_comm);
    if (rank == 0)
      printf("PMPI: %e\n", time);


    /**************************************
    **** Hierarchical Alltoall Timings ****
    **************************************/
    // 1. Full hierarchical pairwise alltoall
    time = test_collective(alltoall_hierarchical_pairwise, true, sendbuf, s, sendtype, recvbuf, s, recvtype, comm);
    if (rank == 0) printf("Hierarchical Pairwise: %e\n", time);

    // 2. Full hierarchical nonblocking alltoall
    time = test_collective(alltoall_hierarchical_nonblocking, true, sendbuf, s, sendtype, recvbuf, s, recvtype, comm);
    if (rank == 0) printf("Hierarchical Nonblocking: %e\n", time);

    // 3: Gather to rank 0 on local comm
    time = test_collective(MPI_Gather, true, sendbuf, s*num_procs, sendtype, local_sendbuf, s*num_procs, sendtype, 0, comm->local_comm);
    if (rank == 0) printf("Hierarchical Internal: Gather: %e\n", time);

    // 4. Scatter from rank 0 on local comm
    time = test_collective(MPI_Scatter, true, local_recvbuf, s*num_procs, sendtype, recvbuf, s*num_procs, sendtype, 0, comm->local_comm);
    if (rank == 0) printf("Hierarchical Internal: Scatter: %e\n", time);

    // 5. Pairwise alltoall between leaders on group comm
    MPIX_Comm_tag(comm, &tag);
    time = test_collective(pairwise_helper, local_rank == 0, local_sendbuf, s*ppn*ppn, sendtype, local_recvbuf, s*ppn*ppn, recvtype, comm->group_comm, tag);
    if (rank == 0) printf("Hierarchical Internal: Pairwise inter-node alltoall: %e\n", time);

    // 6. Nonblocking alltoall between leaders on group comm
    MPIX_Comm_tag(comm, &tag);
    time = test_collective(nonblocking_helper, local_rank == 0, local_sendbuf, s*ppn*ppn, sendtype, local_recvbuf, s*ppn*ppn, recvtype, comm->group_comm, tag);
    if (rank == 0) printf("Hierarchical Interal: Nonblocking inter-node alltoall: %e\n", time);

    if (local_rank == 0)
    {
      free(local_sendbuf);
      free(local_recvbuf);
    }

    /**************************************
    **** Node-Aware Alltoall Timings ****
    **************************************/
    // 1. Full pairwise node-aware alltoall
    time = test_collective(alltoall_node_aware_pairwise, true, sendbuf, s, sendtype, recvbuf, s, recvtype, comm);
    if (rank == 0) printf("Node-Aware Pairwise: %e\n", time);
  
    // 2. Full nonblocking node-aware alltoall
    time = test_collective(alltoall_node_aware_nonblocking, true, sendbuf, s, sendtype, recvbuf, s, recvtype, comm);
    if (rank == 0) printf("Node-Aware Nonblocking: %e\n", time);

    // 3. Pairwise intra-node alltoall
    MPIX_Comm_tag(comm, &tag);
    time = test_collective(pairwise_helper, true, sendbuf, s*n_nodes, sendtype, recvbuf, s*n_nodes, recvtype, comm->local_comm, tag);
    if (rank == 0) printf("Node-Aware Internal: Pairwise intra-node alltoall: %e\n", time);

    // 4. Pairwise inter-node alltoall
    MPIX_Comm_tag(comm, &tag);
    time = test_collective(pairwise_helper, true, sendbuf, s*ppn, sendtype, recvbuf, s*ppn, recvtype, comm->group_comm, tag);
    if (rank == 0) printf("Node-Aware Internal: Pairwise inter-node alltoall: %e\n", time);

    // 5. Nonblocking intra-node alltoall
    MPIX_Comm_tag(comm, &tag);
    time = test_collective(nonblocking_helper, true, sendbuf, s*n_nodes, sendtype, recvbuf, s*n_nodes, recvtype, comm->local_comm, tag);
    if (rank == 0) printf("Node-Aware Internal: Nonblocking intra-node alltoall: %e\n", time);

    // 6. Nonblocking inter-node alltoall
    MPIX_Comm_tag(comm, &tag);
    time = test_collective(nonblocking_helper, true, sendbuf, s*ppn, sendtype, recvbuf, s*ppn, recvtype, comm->group_comm, tag);
    if (rank == 0) printf("Node-Aware Internal: Nonblocking inter-node alltoall: %e\n", time);

    /**************************************
    **** Multi-leader loop: 4, 8, 16   ****
    **************************************/
    std::vector<int> procs_per_leader_list = {4, 8, 16};
    for (int ctr = 0; ctr < procs_per_leader_list.size(); ctr++)
    {
      int n_procs = procs_per_leader_list[ctr];
      if (ppn < n_procs)
         break;

      if (comm->leader_comm != MPI_COMM_NULL)
          MPIX_Comm_leader_free(comm);

      MPIX_Comm_leader_init(comm, n_procs);

      int leader_rank, leader_size;
      MPI_Comm_rank(comm->leader_comm, &leader_rank);
      MPI_Comm_size(comm->leader_comm, &leader_size);

      int my_leader, num_leaders;
      MPI_Comm_rank(comm->leader_group_comm, &my_leader);
      MPI_Comm_size(comm->leader_group_comm, &num_leaders);
      int leaders_per_node = num_leaders / n_nodes;
if (rank == 0) printf("Nnodes %d, Num leaders %d, leaders per node %d, procs per leader %d\n", n_nodes, num_leaders, leaders_per_node, n_procs);

      if (leader_rank == 0)
      {
        local_sendbuf = (T*)malloc(s * num_procs * leader_size * sizeof(T));
        local_recvbuf = (T*)malloc(s * num_procs * leader_size * sizeof(T));
      }
      /**************************************
      **** Multi-leader Alltoall Timings ****
      **************************************/
      // 1. Full multileader pairwise alltoall
      time = test_collective(alltoall_multileader, true, pairwise_helper, sendbuf, s, sendtype, recvbuf, s, recvtype, comm, leaders_per_node);
      if (rank == 0) printf("Hierarchical Pairwise (N Leaders %d): %e\n", leaders_per_node, time);
int new_num_leaders;
MPI_Comm_size(comm->leader_comm, &new_num_leaders);
if (rank == 0) printf("Num Leaders %d, new_num_leaders %d\n", num_leaders, new_num_leaders);
      // 2. Full multileader nonblocking alltoall
      time = test_collective(alltoall_multileader, true, nonblocking_helper, sendbuf, s, sendtype, recvbuf, s, recvtype, comm, leaders_per_node);
      if (rank == 0) printf("Hierarchical Nonblocking (N Leaders %d): %e\n", leaders_per_node, time);

      // 3. Gather to rank 0 on leader comm
      time = test_collective(MPI_Gather, true, sendbuf, s*num_procs, sendtype, local_sendbuf, s*num_procs, sendtype, 0, comm->leader_comm);
      if (rank == 0) printf("Hierarchical (N Leaders %d) Internal: Gather: %e\n", leaders_per_node, time);

      // 4. Scatter from rank 0 on leader comm
      time = test_collective(MPI_Scatter, true, local_recvbuf, s*num_procs, sendtype, recvbuf, s*num_procs, sendtype, 0, comm->leader_comm);
      if (rank == 0) printf("Hierarchical (N Leaders %d) Internal: Scatter: %e\n", leaders_per_node, time);

      // 5. Pairwise alltoall between leaders on leader_group_comm
      time = test_collective(pairwise_helper, leader_rank == 0, local_sendbuf, s*leader_size*leader_size, sendtype, local_recvbuf, s*leader_size*leader_size, recvtype, comm->leader_group_comm, tag);
      if (rank == 0) printf("Hierarchical (N Leaders %d) Internal: Alltoall Pairwise: %e\n", num_leaders, time);

      // 6. Nonblocking alltoall between leaders on leader_group_comm
      time = test_collective(nonblocking_helper, leader_rank == 0, local_sendbuf, s*leader_size*leader_size, sendtype, local_recvbuf, s*leader_size*leader_size, recvtype, comm->leader_group_comm, tag);
      if (rank == 0) printf("Hierarchical (N Leaders %d) Internal: Alltoall Nonblocking: %e\n", num_leaders, time);

      if (leader_rank == 0)
      {
        free(local_sendbuf);
        free(local_recvbuf);
      }

      /**************************************
      **** Node-Aware Alltoall Timings ****
      **************************************/
      // 1. Full pairwise locality-aware alltoall
      time = test_collective(alltoall_locality_aware, true, pairwise_helper, sendbuf, s, sendtype, recvbuf, s, recvtype, comm, leaders_per_node);
      if (rank == 0) printf("Locality-Aware (N Groups %d) Alltoall Pairwise: %e\n", leaders_per_node, time);

      // 2. Full nonblocking locality-aware alltoall
      time = test_collective(alltoall_locality_aware, true, nonblocking_helper, sendbuf, s, sendtype, recvbuf, s, recvtype, comm, leaders_per_node);
      if (rank == 0) printf("Locality-Aware (N Groups %d) Alltoall Nonblocking: %e\n", leaders_per_node, time);

      // 3. Pairwise Intra-node alltoall
      MPIX_Comm_tag(comm, &tag);
      time = test_collective(pairwise_helper, true, sendbuf, s*leader_size, sendtype, recvbuf, s*leader_size, recvtype, comm->leader_comm, tag);
      if (rank == 0) printf("Locality-Aware (N Groups %d) Internal: Intra-node pairwise alltoall: %e\n", leaders_per_node, time);

      // 4. Pairwise Inter-node alltoall
      MPIX_Comm_tag(comm, &tag);
      time = test_collective(pairwise_helper, true, sendbuf, s*leaders_per_node, sendtype, recvbuf, s*leaders_per_node, recvtype, comm->leader_group_comm, tag);
      if (rank == 0) printf("Locality-Aware (N Groups %d) Internal: Inter-node pairwise alltoall: %e\n", leaders_per_node, time);

      // 3. Nonblocking Intra-node alltoall
      MPIX_Comm_tag(comm, &tag);
      time = test_collective(nonblocking_helper, true, sendbuf, s*leader_size, sendtype, recvbuf, s*leader_size, recvtype, comm->leader_comm, tag);
      if (rank == 0) printf("Locality-Aware (N Groups %d) Internal: Intra-node nonblocking alltoall: %e\n", leaders_per_node, time);

      // 4. Nonblocking Inter-node alltoall
      MPIX_Comm_tag(comm, &tag);
      time = test_collective(nonblocking_helper, true, sendbuf, s*leaders_per_node, sendtype, recvbuf, s*leaders_per_node, recvtype, comm->leader_group_comm, tag);
      if (rank == 0) printf("Locality-Aware (N Groups %d) Internal: Inter-node nonblocking alltoall: %e\n", leaders_per_node, time);

      MPIX_Comm_leader_free(comm);

    }
  }
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int max_p = 12;
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
