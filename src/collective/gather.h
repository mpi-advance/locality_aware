#include "collective.h"

// TODO : Currently root is always 0
int gather(const void* sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void* recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        int root,
        MPI_Comm comm);
