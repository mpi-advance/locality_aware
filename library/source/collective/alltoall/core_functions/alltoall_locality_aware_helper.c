#include <string.h>

#include "collective/alltoall.h"
#include "locality_aware.h"


int alltoall_locality_aware_helper(alltoall_helper_ftn f,
                                   const void* sendbuf,
                                   const int sendcount,
                                   MPI_Datatype sendtype,
                                   void* recvbuf,
                                   const int recvcount,
                                   MPI_Datatype recvtype,
                                   MPIL_Comm* comm,
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

    char* recv_buffer = (char*)recvbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    int n_groups = num_procs / ppg;

    char* tmpbuf = (char*)malloc(num_procs * sendcount * send_size);

    // 1. Alltoall between group_comms (all data for any process on node)
    f(sendbuf,
      ppg * sendcount,
      sendtype,
      tmpbuf,
      ppg * recvcount,
      recvtype,
      group_comm,
      tag);

    // 2. Re-pack
    int ctr = 0;
    for (int dest_proc = 0; dest_proc < ppg; dest_proc++)
    {
        int offset = dest_proc * recvcount * recv_size;
        for (int origin = 0; origin < n_groups; origin++)
        {
            int node_offset = origin * ppg * recvcount * recv_size;
            memcpy(&(recv_buffer[ctr]),
                   &(tmpbuf[node_offset + offset]),
                   recvcount * recv_size);
            ctr += recvcount * recv_size;
        }
    }

    // 3. Local alltoall
    f(recvbuf,
      n_groups * recvcount,
      recvtype,
      tmpbuf,
      n_groups * recvcount,
      recvtype,
      local_comm,
      tag);

    // 4. Re-order
    ctr = 0;
    for (int node = 0; node < n_groups; node++)
    {
        int node_offset = node * recvcount * recv_size;
        for (int dest = 0; dest < ppg; dest++)
        {
            int dest_offset = dest * n_groups * recvcount * recv_size;
            memcpy(&(recv_buffer[ctr]),
                   &(tmpbuf[node_offset + dest_offset]),
                   recvcount * recv_size);
            ctr += recvcount * recv_size;
        }
    }

    free(tmpbuf);
    return MPI_SUCCESS;
}
