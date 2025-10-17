#include <stdlib.h>
#include <string.h>

#include "communicator/MPIL_Comm.h"
#include "communicator/MPIL_Info.h"
#include "locality_aware.h"
#include "neighborhood/MPIL_Topo.h"
#include "neighborhood/alltoall_crs.h"

// Non-persistent, locality-aware == call dynamic version
int neighbor_alltoallv_locality(const void* sendbuf,
                                const int sendcounts[],
                                const int sdispls[],
                                MPI_Datatype sendtype,
                                void* recvbuf,
                                const int recvcounts[],
                                const int rdispls[],
                                MPI_Datatype recvtype,
                                MPIL_Topo* topo,
                                MPIL_Comm* comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_nnz  = topo->outdegree;
    int send_size = 0;
    for (int i = 0; i < send_nnz; i++)
    {
        send_size += sendcounts[i];
    }

    int recv_nnz, recv_size;
    int *src_tmp, *recvcounts_tmp, *rdispls_tmp;
    char* recvvals_tmp;

    int recv_bytes;
    MPI_Type_size(recvtype, &recv_bytes);

    MPIL_Info* xinfo;
    MPIL_Info_init(&xinfo);

    alltoallv_crs_personalized(send_nnz,
                               send_size,
                               topo->destinations,
                               sendcounts,
                               sdispls,
                               sendtype,
                               sendbuf,
                               &recv_nnz,
                               &recv_size,
                               &src_tmp,
                               &recvcounts_tmp,
                               &rdispls_tmp,
                               recvtype,
                               (void**)&recvvals_tmp,
                               xinfo,
                               comm);

    char* recvvals = (char*)recvbuf;

    int idx, proc;
    int* new_proc_idx = (int*)malloc(num_procs * sizeof(int));
    for (int i = 0; i < recv_nnz; i++)
    {
        new_proc_idx[src_tmp[i]] = i;
    }
    for (int i = 0; i < recv_nnz; i++)
    {
        proc = topo->sources[i];
        idx  = new_proc_idx[proc];
        memcpy(&(recvvals[rdispls[i] * recv_bytes]),
               &(recvvals_tmp[rdispls_tmp[idx] * recv_bytes]),
               recvcounts[i] * recv_bytes);
    }
    free(new_proc_idx);

    MPIL_Info_free(&xinfo);

    MPIL_Free(src_tmp);
    MPIL_Free(recvcounts_tmp);
    MPIL_Free(rdispls_tmp);
    MPIL_Free(recvvals_tmp);

    return MPI_SUCCESS;
}
