#include "heterogeneous/gpu_allreduce.h"
#include "heterogeneous/gpu_utils.h"
#include "collective/allreduce.hpp"
#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"

// ASSUMES 1 CPU CORE PER GPU (Standard for applications)
int gpu_aware_allreduce(allreduce_helper_ftn f,
                        const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm)
{
    return allreduce_impl(f, sendbuf, recvbuf, count, 
            datatype, op, comm, MPIL_GPU_Alloc, MPIL_GPU_Free);
}

int gpu_aware_allreduce_recursive_doubling(const void* sendbuf,
                                           void* recvbuf,
                                           int count,
                                           MPI_Datatype datatype,
                                           MPI_Op op,
                                           MPIL_Comm* comm)
{
    return gpu_aware_allreduce(allreduce_recursive_doubling_helper,
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
    return gpu_aware_allreduce(allreduce_dissemination_loc_helper,
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
    return gpu_aware_allreduce(allreduce_dissemination_ml_helper,
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
    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm->global_comm);
}


int copy_to_cpu_allreduce(allreduce_helper_ftn f,
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

    gpuDeviceSynchronize();

    void* cpu_sendbuf = malloc(count*type_size);
    void* cpu_recvbuf = malloc(count*type_size);

    memcpy(cpu_sendbuf, sendbuf, count*type_size);
    gpuDeviceSynchronize();

    //ierr += gpuMemcpy(cpu_sendbuf, sendbuf, count*type_size, gpuMemcpyDeviceToHost);

    ierr += allreduce_impl(f, cpu_sendbuf, cpu_recvbuf, count, datatype, op, comm,
                    MPIL_Alloc, MPIL_Free);

    gpuDeviceSynchronize();
    memcpy(recvbuf, cpu_recvbuf, count*type_size);

    //ierr += gpuMemcpy(recvbuf, cpu_recvbuf, count*type_size, gpuMemcpyHostToDevice);

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
    return copy_to_cpu_allreduce(allreduce_recursive_doubling_helper,
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
    return copy_to_cpu_allreduce(allreduce_dissemination_loc_helper,
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
    return copy_to_cpu_allreduce(allreduce_dissemination_ml_helper,
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
    int ierr = 0;

    int type_size;
    MPI_Type_size(datatype, &type_size);

    void* cpu_sendbuf = malloc(count*type_size);
    void* cpu_recvbuf = malloc(count*type_size);

    memcpy(cpu_sendbuf, sendbuf, count*type_size);

    ierr += MPI_Allreduce(cpu_sendbuf, cpu_recvbuf, count, datatype, op,
                    comm->global_comm);

    memcpy(recvbuf, cpu_recvbuf, count*type_size);

    free(cpu_sendbuf);
    free(cpu_recvbuf);

    return ierr;   
}
