#include "collective/allgather.h"
#include <math.h>
// Implements the Bruck allgather algorithm
// Note: current implementation will not work for non-contiguous datatypes
// To work with non-contig data, need to get extent for recvbuf to index into it
int allgather_bruck(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int tag;
    get_tag(comm, &tag);

    int bytes;
    MPI_Type_size(recvtype, &bytes);
    int count_bytes = recvcount * bytes;

    char* _recvbuf = (char*)recvbuf;
    char* tmpbuf = (char*)malloc(count_bytes * num_procs);

    // Sendrecv instead of memcpy, so that it works on the GPUs
    MPI_Sendrecv(sendbuf, sendcount, sendtype, rank, tag, 
            tmpbuf, recvcount, recvtype, rank, tag, 
            comm->global_comm, MPI_STATUS_IGNORE);

    int log_procs = (int)log2(num_procs);
    int log2_num_procs = 1 << log_procs;    

    int pow_i = 1;
    int send_proc, recv_proc;
    for (int i = 0; i < log_procs; i++)
    {
        send_proc = (rank + pow_i) % num_procs;
        recv_proc = (rank - pow_i + num_procs) % num_procs;
        MPI_Sendrecv(tmpbuf, recvcount * pow_i, recvtype, send_proc, tag,
                tmpbuf + pow_i * count_bytes, recvcount * pow_i, recvtype,
                recv_proc, tag, comm->global_comm, MPI_STATUS_IGNORE);
        pow_i *= 2;
    }

    // If non-power-of-2 process count, 1 more step
    if (log2_num_procs != num_procs)
    {
        int count = num_procs - log2_num_procs;
        send_proc = (rank + pow_i) % num_procs;
        recv_proc = (rank - pow_i + num_procs) % num_procs;
        MPI_Sendrecv(tmpbuf, recvcount * count, recvtype, send_proc, tag,
                tmpbuf + pow_i * count_bytes, recvcount * count, recvtype,
                recv_proc, tag, comm->global_comm, MPI_STATUS_IGNORE);
    }

    int n_first_group = num_procs - rank;
    int n_last_group = num_procs - n_first_group;

    // Sendrecvs instead of memcpys, so that it works on the GPUs    
    MPI_Sendrecv(tmpbuf, recvcount * n_first_group, recvtype, rank, tag,
            _recvbuf + rank * count_bytes, recvcount * n_first_group,
            recvtype, rank, tag, comm->global_comm, MPI_STATUS_IGNORE);
    if (rank != 0)
        MPI_Sendrecv(tmpbuf + (n_first_group * count_bytes), recvcount * n_last_group, 
                recvtype, rank, tag,
                _recvbuf, recvcount * n_last_group, recvtype, rank, tag,
                comm->global_comm, MPI_STATUS_IGNORE);
    
    free(tmpbuf);

    return MPI_SUCCESS;
}
