#include "../../../../include/collective/alltoall.h"

#include <math.h>
#include <string.h>

/* #ifdef GPU
#include "../../include/heterogenous/gpu_alltoall.h"
#endif */



int alltoall_locality_aware_pairwise(const void* sendbuf,
                                     const int sendcount,
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcount,
                                     MPI_Datatype recvtype,
                                     MPIL_Comm* comm)
{
    return alltoall_locality_aware(pairwise_helper,
                                   sendbuf,
                                   sendcount,
                                   sendtype,
                                   recvbuf,
                                   recvcount,
                                   recvtype,
                                   comm,
                                   4);
}
