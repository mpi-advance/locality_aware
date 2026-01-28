#include "heterogeneous/gpu_allreduce.h"
#include "heterogeneous/gpu_utils.h"
#include "collective/allreduce_init.h"
#include "communicator/MPIL_Comm.h"
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

#if defined(MPI4)
int gpu_aware_allreduce_pmpi_init(const void* sendbuf,
                        void* recvbuf,
                        int count,
                        MPI_Datatype datatype,
                        MPI_Op op,
                        MPIL_Comm* comm,
                        MPIL_Info* info,
                        MPIL_Request** req_ptr)
{
    MPIL_Request* request;
    init_request(&request);
    allocate_requests(1, &(request->global_requests));
    *req_ptr = request;
    request->start_function = pmpi_start;
    request->wait_function = pmpi_wait;

    return PMPI_Allreduce_init(sendbuf, recvbuf, count, datatype, op, 
            comm->global_comm, MPI_INFO_NULL, &(request->global_requests[0]));

}
#endif


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
    void* cpu_sendbuf = malloc(count*type_size);
    void* cpu_recvbuf = malloc(count*type_size);

#if defined(APU)
    memcpy(cpu_sendbuf, sendbuf, count*type_size);
#else
    gpuMemcpy(cpu_sendbuf, sendbuf, count*type_size, gpuMemcpyDeviceToHost);
    gpuStreamSynchronize(0);
#endif 

    ierr += f(cpu_sendbuf, cpu_recvbuf, count, datatype, op, comm,
                    info, req_ptr, MPIL_Alloc, MPIL_Free);

#if defined(APU)
    memcpy(recvbuf, cpu_recvbuf, count*type_size);
#else
    gpuMemcpy(recvbuf, cpu_recvbuf, count*type_size, gpuMemcpyHostToDevice);
    gpuStreamSynchronize(0);
#endif

    MPIL_Request* request = *req_ptr;
    request->cpu_sendbuf = cpu_sendbuf;
    request->cpu_recvbuf = cpu_recvbuf;
    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;

    gpuDeviceSynchronize();

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

#if defined(MPI4)
int copy_to_cpu_allreduce_pmpi_init(const void* sendbuf,
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

    request->cpu_sendbuf = malloc(count*type_size);
    request->cpu_recvbuf = malloc(count*type_size);

#if defined(APU)
    memcpy(cpu_sendbuf, sendbuf, count*type_size);
#else
    gpuMemcpy(cpu_sendbuf, sendbuf, count*type_size, gpuMemcpyDeviceToHost);
    gpuStreamSynchronize(0); // needed on tuolumne
#endif

    ierr += PMPI_Allreduce_init(cpu_sendbuf, cpu_recvbuf, count, datatype, op,
                    comm->global_comm, info, &(request->global_requests[0]));

#if defined(APU)
    memcpy(recvbuf, cpu_recvbuf, count*type_size);
#else
    gpuMemcpy(recvbuf, cpu_recvbuf, count*type_size, gpuMemcpyHostToDevice);
    gpuStreamSynchronize(0); // needed on tuolumne
#endif

    free(cpu_sendbuf);
    free(cpu_recvbuf);

    return ierr;   
}
#endif
