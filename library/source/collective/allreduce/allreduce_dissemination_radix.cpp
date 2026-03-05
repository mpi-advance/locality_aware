#include "collective/allreduce.h"
#include "locality_aware.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

// TODO: fix memset to allow for gpus
// Warning: assumes even numbers of processes per node
int allreduce_dissemination_radix(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm)
{
    if (count == 0)
        return MPI_SUCCESS;

    return allreduce_dissemination_radix_helper(sendbuf, recvbuf, count,
            datatype, op, comm, MPIL_Alloc, MPIL_Free);
}

int allreduce_dissemination_radix_helper(const void* sendbuf,
                                         void* recvbuf,
                                         int count,
                                         MPI_Datatype datatype,
                                         MPI_Op op,
                                         MPIL_Comm* comm,
                                         MPIL_Alloc_ftn alloc_ftn, 
                                         MPIL_Free_ftn free_ftn)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int radix = mpil_collective_radix;

    int tag;
    get_tag(comm, &tag);

    return allreduce_dissemination_radix_core(
                   sendbuf, recvbuf, count, datatype, op, 
                   comm, tag, radix,
                   alloc_ftn, free_ftn);
}

int allreduce_dissemination_radix_core(
                        const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm, 
                        int tag,
                        int radix,
                        MPIL_Alloc_ftn alloc_ftn,
                        MPIL_Free_ftn free_ftn)
{
    int type_size;
    MPI_Type_size(datatype, &type_size);

    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    // Send `sendbuf` into `recvbuf` (Sendrecv to work on CPU or GPU)
    MPI_Sendrecv(sendbuf, count, datatype, rank, tag, 
            recvbuf, count, datatype, rank, tag, comm->global_comm,
            MPI_STATUS_IGNORE);

    int pow_radix_num_procs = 1;
    while (pow_radix_num_procs * radix <= num_procs)
        pow_radix_num_procs *= radix;
    int mult = num_procs / pow_radix_num_procs;
    int max_proc = mult * pow_radix_num_procs;
    int extra = num_procs - max_proc;

    MPI_Request* request = (MPI_Request*)malloc(2*radix*sizeof(MPI_Request));

    char *tmpbuf;
    alloc_ftn((void**)(&tmpbuf), radix*type_size*count);

    if (rank >= max_proc)
    {
        int proc = rank - max_proc;
        MPI_Send(recvbuf, count, datatype, proc, tag, comm->global_comm);
        MPI_Recv(recvbuf, count, datatype, proc, tag, comm->global_comm, 
                MPI_STATUS_IGNORE);
    }
    else
    {
        if (rank < extra)
        {
            MPI_Recv(tmpbuf, count, datatype, max_proc + rank, tag,
                   comm->global_comm, MPI_STATUS_IGNORE);
            MPI_Reduce_local(tmpbuf, recvbuf, count, datatype, op);
        }

        for (int stride_start = 1; stride_start < max_proc; stride_start *= radix)
        {
            int n_msgs = 0;
            for (int step = 1; step < radix; step++)
            {
                int stride = stride_start * step;
                if (stride < max_proc)
                {
                    int send_proc = (rank - stride + max_proc) % max_proc;
                    int recv_proc = (rank + stride) % max_proc;
                    MPI_Isend(recvbuf, count, datatype, send_proc, tag,
                            comm->global_comm, &(request[n_msgs++]));
                    MPI_Irecv(tmpbuf + (step-1)*count*type_size, count, datatype, recv_proc, tag,
                            comm->global_comm, &(request[n_msgs++]));
                }
            }
            MPI_Waitall(n_msgs, request, MPI_STATUSES_IGNORE);
            for (int step = 1; step < radix; step++)
            {
                int stride = stride_start * step;
                if (stride < max_proc)
                    MPI_Reduce_local(tmpbuf+(step-1)*count*type_size, recvbuf, count,
                            datatype, op);
            }
        }


        if (rank < extra)
        {
            MPI_Send(recvbuf, count, datatype, max_proc + rank, tag, comm->global_comm);
        }
    }

    free(request);
    free_ftn(tmpbuf);

    return MPI_SUCCESS;
}


