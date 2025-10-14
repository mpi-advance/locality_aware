#include "collective/alltoall.h"

//#include <math.h>
//#include <string.h>






int alltoall_locality_aware(alltoall_helper_ftn f,
                            const void* sendbuf,
                            const int sendcount,
                            MPI_Datatype sendtype,
                            void* recvbuf,
                            const int recvcount,
                            MPI_Datatype recvtype,
                            MPIL_Comm* comm,
                            int groups_per_node)
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

    if (groups_per_node > 1)
    {
        if (ppn < groups_per_node)
        {
            groups_per_node = ppn;
        }
        int procs_per_group = ppn / groups_per_node;

        if (comm->leader_comm != MPI_COMM_NULL)
        {
            int ppg;
            MPI_Comm_size(comm->leader_comm, &ppg);
            if (ppg != procs_per_group)
            {
                MPI_Comm_free(&(comm->leader_comm));
            }
        }

        if (comm->leader_comm == MPI_COMM_NULL)
        {
            MPIL_Comm_leader_init(comm, procs_per_group);
        }

        local_comm = comm->leader_comm;
        group_comm = comm->leader_group_comm;
    }

    return alltoall_locality_aware_helper(f,
                                          sendbuf,
                                          sendcount,
                                          sendtype,
                                          recvbuf,
                                          recvcount,
                                          recvtype,
                                          comm,
                                          groups_per_node,
                                          local_comm,
                                          group_comm,
                                          tag);
}
