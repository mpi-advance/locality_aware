#include "bcast.h"
#include <math.h>

// TODO : currently root is always 0
int bcast(void* buffer,
        int count,
        MPI_Datatype datatype,
        int root,
        MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int num_steps = log2(num_procs);
    int tag = 204857;
    MPI_Status status;

    int stride = num_procs / 2;
    for (int i = 0; i < num_steps; i++)
    {
        if (rank % (stride*2) == 0)
        {
            // Sending Proc
            MPI_Send(buffer, count, datatype, rank + stride, tag, comm);
        }
        else if (rank % stride == 0)
        {
            // Recving Proc
            MPI_Recv(buffer, count, datatype, rank - stride, tag, comm, &status);
        }

        stride /= 2;
    }
    return 0;
}
