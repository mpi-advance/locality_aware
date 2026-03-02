#include <string.h>

#include "collective/alltoall.h"
#include "locality_aware.h"

int alltoall_multileader_locality(alltoall_helper_ftn f,
                                  const void* sendbuf,
                                  const int sendcount,
                                  MPI_Datatype sendtype,
                                  void* recvbuf,
                                  const int recvcount,
                                  MPI_Datatype recvtype,
                                  MPIL_Comm* comm)
{
    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag;
    get_tag(comm, &tag);

    if (comm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(comm);
    }

    int local_rank, ppn;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &ppn);

    if (comm->leader_comm == MPI_COMM_NULL)
    {
        int num_leaders_per_node = 4;
        if (ppn < num_leaders_per_node)
        {
            num_leaders_per_node = ppn;
        }
        MPIL_Comm_leader_init(comm, ppn / num_leaders_per_node);
    }

    int procs_per_leader, leader_rank;
    MPI_Comm_rank(comm->leader_comm, &leader_rank);
    MPI_Comm_size(comm->leader_comm, &procs_per_leader);

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    // TODO: currently assuming full nodes, even procs_per_leader per node
    //    this is common, so fair assumption for now
    //    likely need to fix before using in something like Trilinos
    int n_nodes   = num_procs / ppn;
    int n_leaders = num_procs / procs_per_leader;

    int leaders_per_node;
    MPI_Comm_size(comm->leader_local_comm, &leaders_per_node);

    char* local_send_buffer = NULL;
    char* local_recv_buffer = NULL;
    if (leader_rank == 0)
    {
        local_send_buffer =
            (char*)malloc(procs_per_leader * num_procs * sendcount * send_size);
        local_recv_buffer =
            (char*)malloc(procs_per_leader * num_procs * recvcount * recv_size);
    }
    else
    {
        local_send_buffer = (char*)malloc(sizeof(char));
        local_recv_buffer = (char*)malloc(sizeof(char));
    }
    // 1. Gather locally
    MPI_Gather(send_buffer,
               sendcount * num_procs,
               sendtype,
               local_recv_buffer,
               sendcount * num_procs,
               sendtype,
               0,
               comm->leader_comm);

    // 2. Re-pack for sends
    // Assumes SMP ordering
    // TODO: allow for other orderings
    int ctr;

    if (leader_rank == 0)
    {
        /* alltoall_locality_aware_helper(f, sendbuf, procs_per_leader*sendcount,
         *   sendtype, recvbuf, procs_per_leader*recvcount, recvtype, comm,
         * groups_per_node, comm->leader_local_comm, comm->group_comm);
         */

        ctr = 0;
        for (int dest_node = 0; dest_node < n_leaders; dest_node++)
        {
            int dest_node_start = dest_node * procs_per_leader * sendcount * send_size;
            for (int origin_proc = 0; origin_proc < procs_per_leader; origin_proc++)
            {
                int origin_proc_start = origin_proc * num_procs * sendcount * send_size;
                memcpy(&(local_send_buffer[ctr]),
                       &(local_recv_buffer[origin_proc_start + dest_node_start]),
                       procs_per_leader * sendcount * send_size);
                ctr += procs_per_leader * sendcount * send_size;
            }
        }

        // 3. MPI_Alltoall between nodes
        f(local_send_buffer,
          ppn * procs_per_leader * sendcount,
          sendtype,
          local_recv_buffer,
          ppn * procs_per_leader * recvcount,
          recvtype,
          comm->group_comm,
          tag);

        // Re-Pack for exchange between local leaders
        ctr = 0;
        for (int local_leader = 0; local_leader < leaders_per_node; local_leader++)
        {
            int leader_start = local_leader * procs_per_leader * procs_per_leader *
                               sendcount * send_size;
            for (int dest_node = 0; dest_node < n_nodes; dest_node++)
            {
                int dest_node_start =
                    dest_node * ppn * procs_per_leader * sendcount * send_size;
                memcpy(&(local_send_buffer[ctr]),
                       &(local_recv_buffer[dest_node_start + leader_start]),
                       procs_per_leader * procs_per_leader * sendcount * send_size);
                ctr += procs_per_leader * procs_per_leader * sendcount * send_size;
            }
        }

        f(local_send_buffer,
          n_nodes * procs_per_leader * procs_per_leader * sendcount,
          sendtype,
          local_recv_buffer,
          n_nodes * procs_per_leader * procs_per_leader * recvcount,
          recvtype,
          comm->leader_local_comm,
          tag);

        ctr = 0;
        for (int dest_proc = 0; dest_proc < procs_per_leader; dest_proc++)
        {
            int dest_proc_start = dest_proc * recvcount * recv_size;

            for (int orig_node = 0; orig_node < n_nodes; orig_node++)
            {
                int orig_node_start = orig_node * procs_per_leader * procs_per_leader *
                                      recvcount * recv_size;

                for (int orig_leader = 0; orig_leader < leaders_per_node; orig_leader++)
                {
                    int orig_leader_start = orig_leader * n_nodes * procs_per_leader *
                                            procs_per_leader * recvcount * recv_size;
                    for (int orig_proc = 0; orig_proc < procs_per_leader; orig_proc++)
                    {
                        int orig_proc_start =
                            orig_proc * procs_per_leader * recvcount * recv_size;
                        int idx = orig_node_start + orig_leader_start + orig_proc_start +
                                  dest_proc_start;
                        memcpy(&(local_send_buffer[ctr]),
                               &(local_recv_buffer[idx]),
                               recvcount * recv_size);
                        ctr += recvcount * recv_size;
                    }
                }
            }
        }
    }

    // 5. Scatter
    MPI_Scatter(local_send_buffer,
                recvcount * num_procs,
                recvtype,
                recv_buffer,
                recvcount * num_procs,
                recvtype,
                0,
                comm->leader_comm);

    free(local_send_buffer);
    free(local_recv_buffer);

    return MPI_SUCCESS;
}
