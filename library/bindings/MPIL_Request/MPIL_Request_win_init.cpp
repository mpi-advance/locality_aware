#include "locality_aware.h"
#include "persistent/MPIL_Request.h"


#ifdef __cplusplus
extern "C" {
#endif
int MPIL_Request_win_init(MPIL_Request* request, void* recvbuf,
        int bytes, int type_bytes, MPI_Comm comm)
{
    request->win_bytes      = bytes;
    request->win_type_bytes = type_bytes;
    MPI_Win_create(recvbuf,
                   request->win_bytes,
                   request->win_type_bytes,
                   MPI_INFO_NULL,
                   comm,
                   &(request->win));

    return MPI_SUCCESS;
}

#ifdef __cplusplus
}
#endif
