#include "collective/allreduce.h"
#include "locality_aware.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

int allreduce_dissemination(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm)
{
    if (count == 0)
        return MPI_SUCCESS;

    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int type_size;
    MPI_Type_size(datatype, &type_size);

    int tag;
    get_tag(comm, &tag);

    if (sendbuf != MPI_IN_PLACE)
       memcpy(recvbuf, sendbuf, count * type_size); 

    void* tmpbuf = malloc(count*type_size); 
    int send_proc, recv_proc;

    int stride;
    int num_steps = (int)log2(num_procs);
    printf("num steps %d\n", num_steps);

    for (int step = 0; step < num_steps; step++)
    {
        stride = 1 << step;
        send_proc = (rank - stride + num_procs) % num_procs;
        recv_proc = (rank + stride) % num_procs;
        MPI_Sendrecv(recvbuf, count, datatype, send_proc, tag,
                tmpbuf, count, datatype, recv_proc, tag, comm->global_comm, 
                MPI_STATUS_IGNORE);
        MPI_Reduce_local(tmpbuf, recvbuf, count, datatype, op);
    }

    printf("Before: Rank %d, recvbuf %d\n", rank, ((int*)(recvbuf))[0]);
    // If non-power-of-two, one additional exchange
    stride = 1 << num_steps;
    if (stride != num_procs)
    {
        send_proc = (rank - stride + 2*num_procs) % num_procs;
        recv_proc = (rank + stride) % num_procs;
        MPI_Sendrecv(sendbuf, count, datatype, send_proc, tag,
                tmpbuf, count, datatype, recv_proc, tag, comm->global_comm, 
                MPI_STATUS_IGNORE);
        MPI_Reduce_local(tmpbuf, recvbuf, count, datatype, op);
    }

    printf("Rank %d, recvbuf %d\n", rank, ((int*)(recvbuf))[0]);

    free(tmpbuf);
    return MPI_SUCCESS;
}

