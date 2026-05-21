#include "heterogeneous/gpu_allreduce.h"
#include "heterogeneous/gpu_utils.h"
#include "collective/allreduce.h"
#include "locality_aware.h"

int gpu_aware_allreduce(allreduce_ftn f,
                        const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm)
{
    return f(sendbuf, recvbuf, count, datatype, op, comm);
}

int gpu_aware_allreduce_recursive_doubling(const void* sendbuf,
                                           void* recvbuf,
                                           int count,
                                           MPI_Datatype datatype,
                                           MPI_Op op,
                                           MPIL_Comm* comm)
{
    return gpu_aware_allreduce(allreduce_recursive_doubling,
                               sendbuf,
                               recvbuf,
                               count,
                               datatype,
                               op,
                               comm);
}

int gpu_aware_allreduce_dissemination_loc(const void* sendbuf,
                                          void* recvbuf,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op,
                                          MPIL_Comm* comm)
{
    return gpu_aware_allreduce(allreduce_dissemination_loc,
                               sendbuf,
                               recvbuf,
                               count,
                               datatype,
                               op,
                               comm);
}

int gpu_aware_allreduce_dissemination_ml(const void* sendbuf,
                                         void* recvbuf,
                                         int count,
                                         MPI_Datatype datatype,
                                         MPI_Op op,
                                         MPIL_Comm* comm)
{
    return gpu_aware_allreduce(allreduce_dissemination_ml,
                               sendbuf,
                               recvbuf,
                               count,
                               datatype,
                               op,
                               comm);
}

int gpu_aware_allreduce_dissemination_radix(const void* sendbuf,
                                         void* recvbuf,
                                         int count,
                                         MPI_Datatype datatype,
                                         MPI_Op op,
                                         MPIL_Comm* comm)
{
    return gpu_aware_allreduce(allreduce_dissemination_radix,
                               sendbuf,
                               recvbuf,
                               count,
                               datatype,
                               op,
                               comm);
}

int gpu_aware_allreduce_pmpi(const void* sendbuf,
                             void* recvbuf,
                             int count,
                             MPI_Datatype datatype,
                             MPI_Op op,
                             MPIL_Comm* comm)
{
    return gpu_aware_allreduce(allreduce_pmpi,
                               sendbuf,
                               recvbuf,
                               count,
                               datatype,
                               op,
                               comm);
}


int copy_to_cpu_allreduce(allreduce_ftn f,
                          const void* sendbuf,
                          void* recvbuf,
                          int count,
                          MPI_Datatype datatype,
                          MPI_Op op,
                          MPIL_Comm* comm)
{
    int ierr = 0;
    
    int type_size;
    MPI_Type_size(datatype, &type_size);

    // gpuMalloc is too expensive for single allreduce
    void* cpu_sendbuf = malloc(count*type_size);
    void* cpu_recvbuf = malloc(count*type_size);

#if defined(APU)
    memcpy(cpu_sendbuf, sendbuf, count*type_size);
#else
    gpuMemcpy(cpu_sendbuf, sendbuf, count*type_size, gpuMemcpyDeviceToHost);
    gpuStreamSynchronize(0);
#endif 

    ierr += f(cpu_sendbuf, cpu_recvbuf, count, datatype, op, comm,
                    MPIL_Alloc, MPIL_Free);

#if defined(APU)
    memcpy(recvbuf, cpu_recvbuf, count*type_size);
#else
    gpuMemcpy(recvbuf, cpu_recvbuf, count*type_size, gpuMemcpyHostToDevice);
    gpuStreamSynchronize(0);
#endif

    free(cpu_sendbuf);
    free(cpu_recvbuf);

    gpuDeviceSynchronize();

    return ierr;
}


int copy_to_cpu_allreduce_recursive_doubling(const void* sendbuf,
                                           void* recvbuf,
                                           int count,
                                           MPI_Datatype datatype,
                                           MPI_Op op,
                                           MPIL_Comm* comm)
{
    return copy_to_cpu_allreduce(allreduce_recursive_doubling,
                               sendbuf,
                               recvbuf,
                               count,
                               datatype,
                               op,
                               comm);
}

int copy_to_cpu_allreduce_dissemination_loc(const void* sendbuf,
                                          void* recvbuf,
                                          int count,
                                          MPI_Datatype datatype,
                                          MPI_Op op,
                                          MPIL_Comm* comm)
{
    return copy_to_cpu_allreduce(allreduce_dissemination_loc,
                               sendbuf,
                               recvbuf,
                               count,
                               datatype,
                               op,
                               comm);
}

int copy_to_cpu_allreduce_dissemination_ml(const void* sendbuf,
                                         void* recvbuf,
                                         int count,
                                         MPI_Datatype datatype,
                                         MPI_Op op,
                                         MPIL_Comm* comm)
{
    return copy_to_cpu_allreduce(allreduce_dissemination_ml,
                               sendbuf,
                               recvbuf,
                               count,
                               datatype,
                               op,
                               comm);
}

int copy_to_cpu_allreduce_dissemination_radix(const void* sendbuf,
                                         void* recvbuf,
                                         int count,
                                         MPI_Datatype datatype,
                                         MPI_Op op,
                                         MPIL_Comm* comm)
{
    return copy_to_cpu_allreduce(allreduce_dissemination_radix,
                               sendbuf,
                               recvbuf,
                               count,
                               datatype,
                               op,
                               comm);
}


int copy_to_cpu_allreduce_pmpi(const void* sendbuf,
                               void* recvbuf,
                               int count,
                               MPI_Datatype datatype,
                               MPI_Op op,
                               MPIL_Comm* comm)
{
    return copy_to_cpu_allreduce(allreduce_pmpi,
                              sendbuf,
                              recvbuf,
                              count,
                              datatype,
                              op,
                              comm);
    }
