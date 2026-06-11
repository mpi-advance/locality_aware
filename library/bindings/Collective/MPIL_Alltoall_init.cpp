#include "collective/alltoall_init.h"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_collective.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Alltoall_init(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* mpi_comm,
                  MPIL_Info* info,
                  MPIL_Request** req_ptr)
{
    if (sendcount == 0) 
    {
        return MPI_SUCCESS;
    }

    alltoall_ftn method;
    bool gpu_aware = false;
    bool copy_to_cpu = false;

    switch (mpil_alltoall_init_implementation)
    {
#if defined(GPU) 
#if defined(GPU_AWARE)
        case ALLTOALL_INIT_GPU_PAIRWISE:
            method = alltoall_init_pairwise;
            gpu_aware = true;
            break;
        case ALLTOALL_INIT_GPU_NONBLOCKING:
            method = alltoall_init_nonblocking;
            gpu_aware = true;
            break;
#if defined(MPI4)
        case ALLTOALL_INIT_GPU_PMPI:
            method = alltoall_init_pmpi;
            gpu_aware = true;
            break;
#endif
#endif
        case ALLTOALL_INIT_CTC_PAIRWISE:
            method = alltoall_init_pairwise;
            copy_to_cpu = true;
            break;
        case ALLTOALL_INIT_CTC_NONBLOCKING:
            method = alltoall_init_nonblocking;
            copy_to_cpu = true;
            break;
#if defined(MPI4)
        case ALLTOALL_INIT_CTC_PMPI:
            method = alltoall_init_pmpi;
            copy_to_cpu = true;
            break;
#endif
#endif
        case ALLTOALL_INIT_PAIRWISE:
            method = alltoall_init_pairwise;
            break;
        case ALLTOALL_INIT_NONBLOCKING:
            method = alltoall_init_nonblocking;
            break;
#if defined(MPI4)
        case ALLTOALL_INIT_PMPI:
            method = alltoall_init_pmpi;
            break;
#endif
        default:
#if defined(MPI4)
            method = alltoall_init_pmpi;
#else
            method = alltoall_init_pairwise;
#endif
            break;
    }

#if defined(GPU)
    if (gpu_aware)
    {
        return gpu_aware_collective(method, sendbuf, sendcount,
                    sendtype, recvbuf, recvcount, recvtype, mpi_comm,
                    info, req_ptr);
    }
    else if (copy_to_cpu)
    {
        return copy_to_cpu_alltoall_init(method, sendbuf, sendcount,
                    sendtype, recvbuf, recvcount, recvtype, mpi_comm,
                    info, req_ptr);
    }
#endif

    return method(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, mpi_comm,
            info, req_ptr);
}

#ifdef __cplusplus
}
#endif
