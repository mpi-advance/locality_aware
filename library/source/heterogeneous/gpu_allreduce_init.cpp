#include "heterogeneous/gpu_allreduce_init.h"
#include "heterogeneous/gpu_utils.h"
#include "collective/allreduce_init.h"
#include "locality_aware.h"

// ASSUMES 1 CPU CORE PER GPU (Standard for applications)
int gpu_aware_allreduce_init(allreduce_init_helper_ftn f,
                             const void* sendbuf,
                             void* recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op,
                             MPIL_Comm* comm,
                             MPIL_Info* info,
                             MPIL_Request** req_ptr)
{
    // Larger sizes, reductions on GPU
    if (count > 16384)
        return f(sendbuf, recvbuf, count, datatype, op, 
                comm, info, req_ptr, MPIL_GPU_Alloc, MPIL_GPU_Free);

    // Small sizes, reductions on CPU
    return f(sendbuf, recvbuf, count, datatype, op, comm,
            info, req_ptr, MPIL_Alloc, MPIL_Free);
}

int gpu_aware_allreduce_recursive_doubling_init(const void* sendbuf,
                                            void* recvbuf,
                                            int count,
                                            MPI_Datatype datatype,
                                            MPI_Op op,
                                            MPIL_Comm* comm,
                                            MPIL_Info* info,
                                            MPIL_Request** req_ptr)
{
    return gpu_aware_allreduce_init(allreduce_recursive_doubling_init_helper,
                               sendbuf, recvbuf, count, datatype, op, comm, 
                               info, req_ptr);
}

int gpu_aware_allreduce_dissemination_loc_init(const void* sendbuf,
                                          void* recvbuf,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op,
                                          MPIL_Comm* comm,
                                          MPIL_Info* info,
                                          MPIL_Request** req_ptr)
{
    return gpu_aware_allreduce_init(allreduce_dissemination_loc_init_helper,
                               sendbuf, recvbuf, count, datatype, op,
                               comm, info, req_ptr);
}

int gpu_aware_allreduce_dissemination_ml_init(const void* sendbuf,
                                              void* recvbuf,
                                              int count,
                                              MPI_Datatype datatype,
                                              MPI_Op op,
                                              MPIL_Comm* comm,
                                              MPIL_Info* info,
                                              MPIL_Request** req_ptr)
{
    return gpu_aware_allreduce_init(allreduce_dissemination_ml_init_helper,
                               sendbuf, recvbuf, count, datatype, op,
                               comm, info, req_ptr);
}

int gpu_aware_allreduce_dissemination_radix_init(const void* sendbuf,
                                              void* recvbuf,
                                              int count,
                                              MPI_Datatype datatype,
                                              MPI_Op op,
                                              MPIL_Comm* comm,
                                              MPIL_Info* info,
                                              MPIL_Request** req_ptr)
{
    return gpu_aware_allreduce_init(allreduce_dissemination_radix_init_helper,
                               sendbuf, recvbuf, count, datatype, op,
                               comm, info, req_ptr);
}


// TODO -- c2c_start, c2c_wait

// TODO -- c2c_start, c2c_wait
int copy_to_cpu_allreduce_init(allreduce_init_helper_ftn f,
                          const void* sendbuf,
                          void* recvbuf,
                          int count,
                          MPI_Datatype datatype,
                          MPI_Op op,
                          MPIL_Comm* comm,
                          MPIL_Info* info,
                          MPIL_Request** req_ptr)
{
    int ierr = 0;
    
    int type_size;
    MPI_Type_size(datatype, &type_size);

    // gpuMalloc is too expensive for single allreduce
    void *cpu_sendbuf, *cpu_recvbuf;
    MPIL_Alloc(&cpu_sendbuf, count*type_size);
    MPIL_Alloc(&cpu_recvbuf, count*type_size);

    ierr += f(cpu_sendbuf, cpu_recvbuf, count, datatype, op, comm,
                    info, req_ptr, MPIL_Alloc, MPIL_Free);

    MPIL_Request* request = *req_ptr;
    request->tmp_gpubuf = cpu_sendbuf;
    request->gpu_sendbuf = sendbuf;
    request->gpu_recvbuf = recvbuf;

    return ierr;
}


int copy_to_cpu_allreduce_recursive_doubling_init(const void* sendbuf,
                                           void* recvbuf,
                                           int count,
                                           MPI_Datatype datatype,
                                           MPI_Op op,
                                           MPIL_Comm* comm,
                                           MPIL_Info* info,
                                           MPIL_Request** req_ptr)
{
    return copy_to_cpu_allreduce_init(allreduce_recursive_doubling_init_helper,
                               sendbuf, recvbuf, count, datatype, op,
                               comm, info, req_ptr);
}

int copy_to_cpu_allreduce_dissemination_loc_init(const void* sendbuf,
                                          void* recvbuf,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op,
                                          MPIL_Comm* comm,
                                          MPIL_Info* info,
                                          MPIL_Request** req_ptr)
{
    return copy_to_cpu_allreduce_init(allreduce_dissemination_loc_init_helper,
                               sendbuf, recvbuf, count, datatype, op,
                               comm, info, req_ptr);
}

int copy_to_cpu_allreduce_dissemination_ml_init(const void* sendbuf,
                                         void* recvbuf,
                                         int count,
                                         MPI_Datatype datatype,
                                         MPI_Op op,
                                         MPIL_Comm* comm,
                                         MPIL_Info* info,
                                         MPIL_Request** req_ptr)
{
    return copy_to_cpu_allreduce_init(allreduce_dissemination_ml_init_helper,
                               sendbuf, recvbuf, count, datatype, op,
                               comm, info, req_ptr);
}

int copy_to_cpu_allreduce_dissemination_radix_init(const void* sendbuf,
                                         void* recvbuf,
                                         int count,
                                         MPI_Datatype datatype,
                                         MPI_Op op,
                                         MPIL_Comm* comm,
                                         MPIL_Info* info,
                                         MPIL_Request** req_ptr)
{
    return copy_to_cpu_allreduce_init(allreduce_dissemination_radix_init_helper,
                               sendbuf, recvbuf, count, datatype, op,
                               comm, info, req_ptr);
}

