#include "collective/allreduce.hpp"
#include "locality_aware.h"
#include <string.h>
#include <math.h>

int allreduce_recursive_doubling(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm)
{
    return allreduce_impl(allreduce_recursive_doubling_helper,
                   sendbuf, recvbuf, count, datatype, op, comm,
                   MPIL_Alloc, MPIL_Free);
}

int allreduce_recursive_doubling_helper(
                        const void* sendbuf,
                        void* tmpbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm)
{
    if (count == 0)
        return MPI_SUCCESS;

    int type_size;
    MPI_Type_size(datatype, &type_size);

    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag;
    get_tag(comm, &tag);

    if (sendbuf != MPI_IN_PLACE)
        MPI_Sendrecv(sendbuf, count, datatype, rank, tag,
                recvbuf, count, datatype, rank, tag, comm->global_comm,
                MPI_STATUS_IGNORE);

    int proc; 
    int log_procs = (int)log2(num_procs);
    int log2_num_procs = 1 << log_procs;
    int extra_procs = num_procs - log2_num_procs;

    if (rank >= log2_num_procs)
    {
        proc = rank - log2_num_procs;
        MPI_Send(recvbuf, count, datatype, rank - log2_num_procs, tag, comm->global_comm);
        MPI_Recv(recvbuf, count, datatype, rank - log2_num_procs, tag, comm->global_comm,
                MPI_STATUS_IGNORE);
    }
    else
    {
        if (rank < extra_procs)
        {
            MPI_Recv(tmpbuf, count, datatype, rank + log2_num_procs, tag, comm->global_comm,
                    MPI_STATUS_IGNORE);
            MPI_Reduce_local(tmpbuf, recvbuf, count, datatype, op);
        }
        for (int stride = 1; stride < log2_num_procs; stride = stride << 1)
        {
            proc = rank ^ stride;
            MPI_Sendrecv(recvbuf, count, datatype, proc, tag,
                    tmpbuf, count, datatype, proc, tag, comm->global_comm, 
                    MPI_STATUS_IGNORE);
            MPI_Reduce_local(tmpbuf, recvbuf, count, datatype, op);
        }
        if (rank < extra_procs)
        {
            MPI_Send(recvbuf, count, datatype, rank + log2_num_procs, tag, comm->global_comm);
        }
    }

    return MPI_SUCCESS;
}
