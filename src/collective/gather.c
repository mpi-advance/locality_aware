#include "gather.h"
#include <string.h>
#include <math.h>
#include "error.h"


// TODO : Currently root is always 0
int gather(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        int root,
        MPI_Comm comm)
{
    int rank, num_procs;
    MPI_ADVANCE_SUCCESS_OR_RETURN(MPI_Comm_rank(comm, &rank));
    MPI_ADVANCE_SUCCESS_OR_RETURN(MPI_Comm_size(comm, &num_procs));

    int recv_size;
    MPI_ADVANCE_SUCCESS_OR_RETURN(MPI_Type_size(recvtype, &recv_size));

    int num_steps = log2(num_procs);   
    int tag = 204857;
    MPI_Status status;

    memcpy(recvbuf, sendbuf, recvcount*recv_size);
    char* recv_buffer = (char*)recvbuf;

    int stride = 1;
    for (int i = 0; i < num_steps; i++)
    {
        if (rank % (stride*2))
        {
            // Sending Proc
            MPI_ADVANCE_SUCCESS_OR_RETURN(MPI_Send(recvbuf, recvcount*stride, recvtype, rank - stride, tag, comm));
            break;
        }
        else
        {
            // Recving Proc
            MPI_ADVANCE_SUCCESS_OR_RETURN(MPI_Recv(&(recv_buffer[recvcount*stride*recv_size]), recvcount*stride, recvtype,
                    rank + stride, tag, comm, &status)); 
        }

        stride *= 2;
    }
    return MPI_SUCCESS;
}

