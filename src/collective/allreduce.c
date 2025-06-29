#include "allreduce.h"

int allreduce_multileader(const void *sendbuf, 
        void *recvbuf,
        const int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm comm, 
        int n_leaders)
{
    int rank, num_procs;
    MPI_Comm_rank(comm.leader_comm, &rank);
    MPI_Comm_size(comm.global_comm, &num_procs);

    int tag;
    MPIX_Comm_tag(&comm, &tag);

    if (comm.local_comm == MPI_COMM_NULL)
    {
        MPIX_Comm_topo_init(&comm);
    }

    int ppn;
    MPI_Comm_size(comm.local_comm, &ppn);

    MPI_Comm local_comm = comm.local_comm;
    MPI_Comm group_comm = comm.group_comm;

    if (n_leaders > 1)
    {
        if (ppn < n_leaders)
        {
            n_leaders = ppn;
        }

        int procs_per_leader = ppn / n_leaders;

        if (comm.leader_comm != MPI_COMM_NULL)
        {
            int ppl;
            MPI_Comm_size(comm.leader_comm, &ppl);
            if (ppl != procs_per_leader)
            {
                MPI_Comm_free(&comm.leader_comm);
            }
        }

        if (comm.leader_comm == MPI_COMM_NULL)
        {
            MPIX_Comm_leader_init(&comm, procs_per_leader);
        }

        local_comm = comm.leader_comm;
        group_comm = comm.leader_group_comm;
    }

    int data_size;
    MPI_Type_size(datatype, &data_size);

    int local_rank, ppl;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &ppl);

    int n_nodes = num_procs / ppl;

    char* local_recv_buff = NULL;
    if (local_rank == 0)
    {
        local_recv_buff = (char*) malloc(count * data_size);
    }
    else{
        local_recv_buff = (char*) malloc(sizeof(char));
    }

    // 1. Reduce locally
    MPI_Reduce(sendbuf, local_recv_buff, count, datatype, op, 0, local_comm);

    if (local_rank == 0)
    {
        // 2. allreduce between leaders
        MPI_Allreduce(local_recv_buff, recvbuf, count, datatype, op, group_comm);
    }

    // 3. Broadcast
    MPI_Bcast(recvbuf, count, datatype, 0, local_comm);

    return MPI_SUCCESS;
}   

int allreduce_hierarchical(const void *sendbuf,
        void *recvbuf,
        const int count,
	MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm comm)
{
  return allreduce_multileader(sendbuf, recvbuf, count, datatype, op, comm, 4);

}

int allreduce_locality_aware_helper(
	const void* sendbuf,
	const int count,
	MPI_Datatype datatype,
	MPI_Op op,
	MPIX_Comm* comm,
	int groups_per_node,
        MPI_Comm local_comm,
	MPI_Comm group_comm,
        int tag)
{
  int rank, num_procs;
  MPI_Comm_rank(comm->global_comm, &rank);
  MPI_Comm_size(comm->global_comm, &num_procs);

  int ppg;
  MPI_Comm_size(local_comm, &ppg);

  int data_size;
  MPI_Type_size(datatype, &data_size);

  int n_groups = num_procs / ppg;

  char* tmpbuf = (char*) malloc(count*data_size);

  int ret_val = MPI_Allreduce(sendbuf, tmpbuf, count, datatype, op, group_comm);

  free(tmpbuf);
  return ret_val;
}
 
int allreduce_locality_aware(const void *sendbuf,
        void *recvbuf,
        const int count,
        MPI_Datatype datatype,
        MPI_Op op,
	MPIX_Comm comm,
	int groups_per_node)
{
  int rank, num_procs;
  MPI_Comm_rank(comm.global_comm, &rank);
  MPI_Comm_size(comm.global_comm, &num_procs);

  int tag;
  MPIX_Comm_tag(&comm, &tag);

  if (comm.local_comm == MPI_COMM_NULL)
  {
    MPIX_Comm_topo_init(&comm);
  }

  int ppn;
  MPI_Comm_size(comm.local_comm, &ppn);

  MPI_Comm local_comm = comm.local_comm;
  MPI_Comm group_comm = comm.group_comm;

  if (groups_per_node > 1)
  {
    if (ppn < groups_per_node)
    {
      groups_per_node = ppn;
    }
    int procs_per_group = ppn / groups_per_node;

    if (comm.leader_comm != MPI_COMM_NULL)
    {
      int ppg;
      MPI_Comm_size(comm.leader_comm, &ppg);
      if (ppg != procs_per_group)
	MPI_Comm_free(&(comm.leader_comm));
    }

    if (comm.leader_comm != MPI_COMM_NULL)
      MPIX_Comm_leader_init(&comm, procs_per_group);

    local_comm = comm.leader_comm;
    group_comm = comm.leader_group_comm;
  }
}

int allreduce_node_aware(const void *sendbuf,
        void *recvbuf,
        const int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm comm)
{
  return allreduce_locality_aware(sendbuf, recvbuf, count, datatype, op, comm, 1);
}

