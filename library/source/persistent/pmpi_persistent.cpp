#include "persistent/pmpi_persistent.h"
#include "heterogeneous/gpu_utils.h"

int pmpi_start(MPIL_Request* request)
{
#if defined(GPU)
if (request->gpu_sendbuf)
{
#if defined(APU)
    memcpy(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends);
#else
    gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends, 
            gpuMemcpyDeviceToHost, 0);
    gpuStreamSynchronize(0);
#endif
}
#endif

    return PMPI_Start(request->requests);
}

int pmpi_wait(MPIL_Request* request, MPI_Status* status)
{
    int ierr = PMPI_Wait(request->requests, status);

#if defined(GPU)
if (request->gpu_recvbuf)
{
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->recvbuf, request->size_recvs);
#else
    gpuMemcpyAsync(request->gpu_recvbuf, request->recvbuf, request->size_recvs, 
            gpuMemcpyHostToDevice, 0);
    gpuStreamSynchronize(0);
#endif
}
#endif

    return ierr;
}
