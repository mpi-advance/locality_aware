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
  int n_iters = 1;
  if (time > 1e-01)
    n_iters = 1;
  else
  {
    if (time > 1e-02)
      n_iters = 10;
    else
      n_iters = 100;
    time = time_collective(collective_func, n_iters, participant, std::forward<Args>(args)...);
    
    n_iters = (0.1 / time) + 1;
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
  return time;
}

template <typename T>
void print_allgathers(int max_p,
                      const T *sendbuf,
                      MPI_Datatype sendtype,
                      T *recvbuf,
                      MPI_Datatype recvtype,
                      MPIX_Comm *comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int local_rank, ppn;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &ppn);

    int n_nodes;
    MPI_Comm_size(comm->group_comm, &n_nodes);

    int max_s = 1 << max_p;
    
    double time;
    for (int i = 0; i < max_p; i++)
    {
        int s = 1 << i;

        T* tmpbuf = (T*) malloc(s * num_procs * sizeof(T));

        if (rank == 0)
            printf("Size %d\n", s);

        // Standard PMPI Allgather (system MPI)
        time = test_collective(PMPI_Allgather, true, sendbuf, s, sendtype, recvbuf, s, recvtype, comm->global_comm);
        if (rank == 0)
            printf("PMPI: %e\n", time);

        // Hierarchical timings
        // 1. Full hierarchical allgather
        time = test_collective(allgather_hierarchical, true, sendbuf, s, sendtype, recvbuf, s, recvtype, *comm);
        if (rank == 0)
            printf("Hierarchical Allgather: %e\n", time);

        // 2. Gather to rank 0 on local_comm
        time = test_collective(MPI_Gather, true, sendbuf, s, sendtype, tmpbuf, s, sendtype, 0, comm->local_comm);
        if (rank == 0)
            printf("Hierarchical Internal: Gather: %e\n", time);

        // 3. Allgather between leaders on group comm
        time = test_collective(MPI_Allgather, local_rank == 0, tmpbuf, ppn * s, sendtype, recvbuf, ppn * s, recvtype, comm->group_comm);
        if (rank == 0) 
            printf("Hierarchical Internal: Allgather: %e\n", time);

        // 4. Broadcast from leaders
        time = test_collective(MPI_Bcast, true, recvbuf, s * num_procs, recvtype, 0, comm->local_comm);
        if (rank == 0)
            printf("Hierarchical Internal: Broadcast: %e\n", time);

        // Node-Aware Allgather Timings
        // 1. Full node-aware allgather
        time = test_collective(allgather_node_aware, true, sendbuf, s, sendtype, recvbuf, s, recvtype, *comm);
        if (rank == 0)
            printf("Node-Aware Allgather: %e\n", time);

        // 2. Intra-node allgather
        time = test_collective(MPI_Allgather, true, sendbuf, s, sendtype, tmpbuf, s, recvtype, comm->group_comm);
        if (rank == 0)
            printf("Node-Aware Internal: Intra-node Allgather: %e\n", time);

        time = test_collective(MPI_Allgather, true, tmpbuf, n_nodes * s, sendtype, recvbuf, s * n_nodes, recvtype, comm->local_comm);
        if (rank == 0)
            printf("Node-Aware Internal: Inter-node Allgather: %e\n", time);

        free(tmpbuf);

        // Multileader loop: 4, 8, 16
        std::vector<int> procs_per_leader_list = {4, 8, 16};
        for (int ctr = 0; ctr < procs_per_leader_list.size(); ctr++)
        {
            int procs_per_leader = procs_per_leader_list[ctr];
            if (ppn < procs_per_leader)
                break;

            if (comm->leader_comm != MPI_COMM_NULL)
                MPIX_Comm_leader_free(comm);

            MPIX_Comm_leader_init(comm, procs_per_leader);

            int leader_rank, leader_size;
            MPI_Comm_rank(comm->leader_comm, &leader_rank);
            MPI_Comm_size(comm->leader_comm, &leader_size);

            int my_leader, num_leaders;
            MPI_Comm_rank(comm->leader_group_comm, &my_leader);
            MPI_Comm_size(comm->leader_group_comm, &num_leaders);
            int leaders_per_node = num_leaders / n_nodes;
            if (rank == 0)
                printf("Running Multileader/Locality-Aware Tests with %d Processes Per Leader, %d Leaders Per Node", procs_per_leader, leaders_per_node);

            tmpbuf = (T*) malloc(s * num_procs * sizeof(T));

            // Multi-leader Allgather Timings
            // 1. Full multileader allgather
            time = test_collective(allgather_multileader, true, sendbuf, s, sendtype, recvbuf, s, recvtype, *comm);
            if (rank == 0)
                printf("Hierarchical (N leaders %d): %e\n", leaders_per_node, time);

            // 2. Gather to rank 0 on leader comm
            time = test_collective(MPI_Gather, true, sendbuf, s, sendtype, tmpbuf, s, sendtype, 0, comm->leader_comm);
            if (rank == 0)
                printf("Hierarchical (N Leaders %d) Internal: Gather: %e\n", leaders_per_node, time);

            // 3. Allgather between leaders on leader group comm
            time = test_collective(MPI_Allgather, leader_rank == 0, tmpbuf, s * leader_size, sendtype, recvbuf, s * leader_size, recvtype, comm->leader_group_comm);
            if (rank == 0)
                printf("Hierarchical (N Leaders %d) Internal: Allgather: %e\n", leaders_per_node, time);

            // 4. Broadcast from leaders
            time = test_collective(MPI_Bcast, true, recvbuf, s * num_procs, recvtype, 0, comm->leader_comm);
            if (rank == 0)
                printf("Hierarchical (N leaders %d) Internal: Broadcast: %e\n", leaders_per_node, time);

            // Locality-Aware Allgather Timings
            // 1. Full locality-aware allgather
            time = test_collective(allgather_locality_aware, true, sendbuf, s, sendtype, recvbuf, s, recvtype, *comm);
            if (rank == 0)
                printf("Locality-Aware (N leaders %d): %e\n", procs_per_leader, time);

            // 2. Intra-node allgather
            time = test_collective(MPI_Allgather, true, sendbuf, s, sendtype, tmpbuf, s, recvtype, comm->leader_group_comm);
            if (rank == 0)
                printf("Locality-Aware (N leaders %d) Internal: Intra-Node Allgather: %e\n", procs_per_leader, time);

            // 3. Inter-node allgather
            time = test_collective(MPI_Allgather, true, tmpbuf, s * procs_per_leader, recvtype, recvbuf, s * procs_per_leader, recvtype, comm->leader_comm);
            if (rank == 0)
                printf("Locality-Aware (N Leaders %d) Internal: Inter-Node Allgather: %e\n", procs_per_leader, time);

            free(tmpbuf);

            // Multileader + Node-Aware Allreduce Timings
            T* local_send_buff = (T*) malloc(s * num_procs * sizeof(T));
            T* local_recv_buff = (T*) malloc(s * num_procs * sizeof(T));

            // 1. Full multileader node-aware allgather
            time = test_collective(allgather_multileader_locality_aware, true, sendbuf, s, sendtype, recvbuf, s, recvtype, *comm);
            if (rank == 0)
                printf("Multileader Node-Aware (N leaders %d) Allgather: %e\n", leaders_per_node, time);

            // 2. Initial Gather
            time = test_collective(MPI_Gather, true, sendbuf, s, sendtype, local_send_buff, s, recvtype, 0, comm->leader_comm);
            if (rank == 0)
                printf("Multileader Node-Aware (N leaders %d) Internal: Gather: %e\n", leaders_per_node, time);

            // 3. Inter-Node Allgather
            time = test_collective(MPI_Allgather, leader_rank == 0, local_send_buff, procs_per_leader * s, sendtype, local_recv_buff, procs_per_leader * s, sendtype, comm->group_comm);
            if (rank == 0)
                printf("Multileader Node-Aware (N Leaders %d) Internal: Internode Allgather: %e\n", leaders_per_node, time);

            // 4. Intra-Node Allgather
            time = test_collective(MPI_Allgather, leader_rank == 0, local_recv_buff, procs_per_leader * n_nodes * s, sendtype, recvbuf, procs_per_leader * n_nodes * s, recvtype, comm->leader_local_comm);
            if (rank == 0)
                printf("Multileader Node-Aware (N Leaders %d) Internal: Intranode Allgather: %e\n", procs_per_leader, time);

            // // 5. Broadcast
            time = test_collective(MPI_Bcast, true, recvbuf, s * num_procs, recvtype, 0, comm->local_comm);
            if (rank == 0)
                printf("Multileader Node-Aware (N Leaders %d) Internal: Broadcast: %e\n", procs_per_leader, time);

            free(local_send_buff);
            free(local_recv_buff);
        }
    }
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int max_p = 11;
  int max_size = 1 << max_p;

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
  
  print_allgathers(max_p, sendbuf.data(), MPI_INT, recvbuf.data(), MPI_INT, xcomm);
}