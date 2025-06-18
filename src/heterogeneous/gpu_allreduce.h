#ifndef MPI_ADVANCE_GPU_ALLREDUCE_H
#define MPI_ADVANCE_GPU_ALLREDUCE_H

#include "collective/collective.h"

#ifdef __cplusplus
extern "C"
{
#endif


int gpu_aware_allreduce(allreduce_ftn f,
        const void* sendbuf,
        void* recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm* comm);
int gpu_aware_allreduce_lane(const void* sendbuf,
        void* recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm* comm);
int gpu_aware_allreduce_loc(const void* sendbuf,
        void* recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm* comm);


#ifdef __cplusplus
}
#endif

#endif
