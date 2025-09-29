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
void print_allreduces(int max_p, 
                      const T *sendbuf, 
                      T *recvbuf, 
                      MPI_Datatype datatype,
                      MPI_Op op,
                      MPIX_Comm *comm, 
                      T *recvbuf_std)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int local_rank, ppn;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &ppn);

    int my_node, n_nodes;
    MPI_Comm_rank(comm->group_comm, &my_node);
    MPI_Comm_size(comm->local_comm, &n_nodes);

    int max_s = 1 << max_p;

    double time;
    
    T* tmpbuf = NULL;

    for (int i = 0; i < max_p; i++)
    {
        int s = 1 << i;

        tmpbuf = (T*) malloc(s * num_procs * ppn * sizeof(T));

        if (rank == 0)
            printf("Size %d\n", s);

        // Standard PMPI Allreduce (system MPI)
        time = test_collective(PMPI_Allreduce, true, sendbuf, recvbuf, s, datatype, op, comm->global_comm);

        if (rank == 0)
            printf("PMPI: %e\n", time);
        
        // Hierarchical timings
        // 1. Full hierarchical pairwise allreduce
        time = test_collective(allreduce_hierarchical, true, sendbuf, recvbuf, s, datatype, op, *comm);
        if (rank == 0)
            printf("Hierarchical Allreduce: %e\n", time);

        // 2. Gather to rank 0 on local comm
        time = test_collective(MPI_Reduce, true, sendbuf, tmpbuf, s, datatype, op, 0, comm->local_comm);
        if (rank == 0)
            printf("Hierarchical Internal: Reduce: %e\n", time);

        // 3. Broacast from rank 0 to local comm
        time = test_collective(MPI_Bcast, true, recvbuf, s, datatype, 0, comm->local_comm);
        if (rank == 0)
            printf("Hierarchical Internal: Brodcast: %e\n", time);

        // 4. Allreduce between leaders on group comm
        time = test_collective(MPI_Allreduce, local_rank == 0, tmpbuf, recvbuf, s, datatype, op, comm->group_comm); 
        if (rank == 0)
            printf("Hierarchical Internal: internode-alltoall: %e\n", time);

        // Node-Aware Allreduce Timings
        // 1. Full node-aware allreduce
        time = test_collective(allreduce_node_aware, true, sendbuf, recvbuf, s, datatype, op, *comm);
        if (rank == 0)
            printf("Node-Aware Allreduce: %e\n", time);

        // 2. Intra-node allreduce
        time = test_collective(MPI_Allreduce, true, sendbuf, tmpbuf, s, datatype, op, comm->group_comm);
        if (rank == 0)
            printf("Node-Aware Internal: Intra-node allreduce: %e\n", time);

        // 3. Inter-node allreduce
        time = test_collective(MPI_Allreduce, true, tmpbuf, recvbuf, s, datatype, op, comm->local_comm);
        if (rank == 0)
            printf("Node-Aware Internal: Inter-node allreduce: %e\n", time);

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
            MPI_Comm_rank(comm->leader_group_comm, &num_leaders);
            int leaders_per_node = num_leaders / n_nodes;
            if (rank == 0)
                printf("Running Multileader/Locality-Aware Tests with %d Processes Per Leader, %d Leaders Per Node\n", procs_per_leader, leaders_per_node);

            tmpbuf = (T*) malloc(s * sizeof(T));

            // Multi-leader Allreduce Timings
            // 1. Full multileader allreduce
            time = test_collective(allreduce_multileader, true, sendbuf, recvbuf, s, datatype, op, *comm);
            if (rank == 0)
                printf("Hierarchical (N leaders %d): %e\n", leaders_per_node, time);

            // 2. Reduce to rank 0 on leader comm
            time = test_collective(MPI_Reduce, true, sendbuf, tmpbuf, s, datatype, op, 0, comm->leader_comm);
            if (rank == 0)
                printf("Hierarchical (N leaders %d) Internal: Reduce: %e\n", leaders_per_node, time);

            // 3. Allreduce between leaders on leader group comm
            time = test_collective(MPI_Allreduce, leader_rank == 0, tmpbuf, recvbuf, s, datatype, op, comm->leader_group_comm);
            if (rank == 0)
                printf("Hierarchical (N Leaders %d) Internal: Allreduce: %e\n", leaders_per_node, time);

            // 4. Broadcast from rank 0 on leader comm
            time = test_collective(MPI_Bcast, true, recvbuf, s, datatype, 0, comm->leader_comm);
            if (rank == 0)
                printf("Hierarchical (N leaders %d) Internal: Broadcast: %e\n", leaders_per_node, time);

            // Locality-Aware Allreduce Timings
            // 1. Full locality-aware allreduce
            time = test_collective(allreduce_locality_aware, true, sendbuf, recvbuf, s, datatype, op, *comm);
            if (rank == 0)
                printf("Locality-Aware (N Groups %d) Allreduce: %e\n", leaders_per_node, time);

            // 2. Intranode allreduce
            time = test_collective(MPI_Allreduce, true, sendbuf, tmpbuf, s, datatype, op, comm->leader_group_comm);
            if (rank == 0)
                printf("Locality-Aware (N Groups %d) Internal: Intra-node allreduce: %e\n", leaders_per_node, time);

            // 3. Internode allreduce
            time = test_collective(MPI_Allreduce, true, tmpbuf, recvbuf, s, datatype, op, comm->leader_comm);
            if (rank == 0)
                printf("Locality-Aware (N Groups %d) Internal: Inter-node allreduce: %e\n", leaders_per_node, time);
            
            free(tmpbuf);
        
            // Multileader + Node Aware Allreduce Timings
            T* local_send_buff = NULL;
            T* local_recv_buff = NULL;
            if (leader_rank == 0)
            {
                local_send_buff = (T*) malloc(s * sizeof(T));
                local_recv_buff = (T*) malloc(s * sizeof(T));
            }
                

            // 1. Full multileader node-aware allreduce
            time = test_collective(allreduce_multileader_locality, true, sendbuf, recvbuf, s, datatype, op, *comm);
            if (rank == 0)
                printf("Multileader Node-Aware (N leaders %d) Allreduce: %e\n", leaders_per_node, time);

            // 2. Initial Reduce
            time = test_collective(MPI_Reduce, true, sendbuf, local_send_buff, s, datatype, op, 0, comm->leader_comm);
            if (rank == 0)
                printf("Multileader Node-Aware (N leaders %d) Internal: MPI_Reduce: %e\n", leaders_per_node, time);

            // 3. Inter-Node Allreduce
            time = test_collective(MPI_Allreduce, leader_rank == 0, local_send_buff, local_recv_buff, s, datatype, op, comm->group_comm);
            if (rank == 0)
                printf("Multileader Node-Aware (N leaders %d) Internal: Inter-Node MPI_Allreduce: %e\n", leaders_per_node, time);

            // 4. Intra-Node Allreduce
            time = test_collective(MPI_Allreduce, leader_rank == 0, local_recv_buff, recvbuf, s, datatype, op, comm->leader_local_comm);
            if (rank == 0)
                printf("Multileader Node-Aware (N leaders %d) Internal: Intra-Node MPI_Allreduce: %e\n", leaders_per_node, time);

            // 5. Broadcast
            time = test_collective(MPI_Bcast, true, recvbuf, s, datatype, 0, comm->leader_comm);
            if (rank == 0)
                printf("Multileader Node-Aware (N leaders %d) Internal: MPI_Bcast: %e\n", leaders_per_node, time);

            if (leader_rank == 0)
            {
                free(local_send_buff);
                free(local_recv_buff);
            }
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
  
  print_allreduces(max_p, sendbuf.data(), recvbuf.data(), MPI_INT, MPI_SUM, xcomm, recvbuf_std.data());
}