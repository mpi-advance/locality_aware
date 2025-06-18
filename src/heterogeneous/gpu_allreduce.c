#include "collective/allreduce.h"
#include "collective/collective.h"
#include "gpu_allreduce.h"

// ASSUMES 1 CPU CORE PER GPU (Standard for applications)
int gpu_aware_allreduce(allreduce_ftn f,
        const void* sendbuf,
        void* recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm* comm)
{
    return f(sendbuf, recvbuf, count, datatype, op, comm);
}

int gpu_aware_allreduce_lane(const void* sendbuf,
        void* recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm* comm)
{
    return gpu_aware_allreduce(allreduce_lane, sendbuf, recvbuf,
            count, datatype, op, comm);
}

int gpu_aware_allreduce_loc(const void* sendbuf,
        void* recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm* comm)
{
    return gpu_aware_allreduce(allreduce_loc, sendbuf, recvbuf, 
            count, datatype, op, comm);
}

