#include "../../../../include/collective/alltoall.h"

//#include <math.h>
#include <string.h>

/* #ifdef GPU
#include "../../include/heterogenous/gpu_alltoall.h"
#endif */

int alltoall_multileader(alltoall_helper_ftn f,
                         const void* sendbuf,
                         const int sendcount,
                         MPI_Datatype sendtype,
                         void* recvbuf,
                         const int recvcount,
                         MPI_Datatype recvtype,
                         MPIL_Comm* comm,
                         int n_leaders)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag;
    MPIL_Comm_tag(comm, &tag);

    if (comm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(comm);
    }

    int ppn;
    MPI_Comm_size(comm->local_comm, &ppn);

    MPI_Comm local_comm = comm->local_comm;
    MPI_Comm group_comm = comm->group_comm;

    if (n_leaders > 1)
    {
        if (ppn < n_leaders)
        {
            n_leaders = ppn;
        }
        int procs_per_leader = ppn / n_leaders;

        // If leader comm exists but with wrong number of leaders per node,
        // free the stale communicator
        if (comm->leader_comm != MPI_COMM_NULL)
        {
            int ppl;
            MPI_Comm_size(comm->leader_comm, &ppl);
            if (ppl != procs_per_leader)
            {
                MPI_Comm_free(&comm->leader_comm);
            }
        }

        // If leader comm does not exist, create it
        if (comm->leader_comm == MPI_COMM_NULL)
        {
            MPIL_Comm_leader_init(comm, procs_per_leader);
        }

        local_comm = comm->leader_comm;
        group_comm = comm->leader_group_comm;
    }

    char* recv_buffer = (char*)recvbuf;
    char* send_buffer = (char*)sendbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    int local_rank, ppl;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &ppl);

    // TODO: currently assuming full nodes, even ppn per node
    //    this is common, so fair assumption for now
    //    likely need to fix before using in something like Trilinos
    int n_nodes = num_procs / ppl;

    char* local_send_buffer = NULL;
    char* local_recv_buffer = NULL;

    if (local_rank == 0)
    {
        local_send_buffer = (char*)malloc(ppl * num_procs * sendcount * send_size);
        local_recv_buffer = (char*)malloc(ppl * num_procs * recvcount * recv_size);
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
               local_comm);

    // 2. Re-pack for sends
    // Assumes SMP ordering
    // TODO: allow for other orderings
    int ctr;

    if (local_rank == 0)
    {
        ctr = 0;
        for (int dest_node = 0; dest_node < n_nodes; dest_node++)
        {
            int dest_node_start = dest_node * ppl * sendcount * send_size;
            for (int origin_proc = 0; origin_proc < ppl; origin_proc++)
            {
                int origin_proc_start = origin_proc * num_procs * sendcount * send_size;
                memcpy(&(local_send_buffer[ctr]),
                       &(local_recv_buffer[origin_proc_start + dest_node_start]),
                       ppl * sendcount * send_size);
                ctr += ppl * sendcount * send_size;
            }
        }

        // 3. MPI_Alltoall between leaders
        f(local_send_buffer,
          ppl * ppl * sendcount,
          sendtype,
          local_recv_buffer,
          ppl * ppl * recvcount,
          recvtype,
          group_comm,
          tag);

        // 4. Re-pack for local scatter
        ctr = 0;
        for (int dest_proc = 0; dest_proc < ppl; dest_proc++)
        {
            int dest_proc_start = dest_proc * recvcount * recv_size;
            for (int orig_proc = 0; orig_proc < num_procs; orig_proc++)
            {
                int orig_proc_start = orig_proc * ppl * recvcount * recv_size;
                memcpy(&(local_send_buffer[ctr]),
                       &(local_recv_buffer[orig_proc_start + dest_proc_start]),
                       recvcount * recv_size);
                ctr += recvcount * recv_size;
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
                local_comm);

    free(local_send_buffer);
    free(local_recv_buffer);

    return MPI_SUCCESS;
}
