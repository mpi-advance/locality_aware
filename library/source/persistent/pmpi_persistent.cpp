#include "persistent/pmpi_persistent.h"
#include "heterogeneous/gpu_utils.h"

int pmpi_start(MPIL_Request* request)
{
#if defined(GPU)
    int gpu_error;
if (request->gpu_sendbuf)
{
#if defined(APU)
    memcpy(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends);
#else
    gpu_error = gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends, 
            gpuMemcpyDeviceToHost, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif

    return PMPI_Start(request->requests);
}

int pmpi_wait(MPIL_Request* request, MPI_Status* status)
{
    int ierr = PMPI_Wait(request->requests, status);

#if defined(GPU)
    int gpu_error;
if (request->gpu_recvbuf)
{
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->recvbuf, request->size_recvs);
#else
    gpu_error = gpuMemcpyAsync(request->gpu_recvbuf, request->recvbuf, request->size_recvs, 
            gpuMemcpyHostToDevice, 0);
    gpu_check(gpu_error);
    gpu_error = gpuStreamSynchronize(0);
    gpu_check(gpu_error);
#endif
}
#endif

    return ierr;
}
