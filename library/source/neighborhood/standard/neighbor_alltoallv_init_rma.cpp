#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"
#include "neighborhood/neighbor.h"
#include "collective/alltoallv_init.h"
#include "string.h"

// Standard, non-persistent neighbor collective
int neighbor_alltoallv_init_rma(const void* sendbuf,
                                     const int sendcounts[],
                                     const int sdispls[],
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcounts[],
                                     const int rdispls[],
                                     MPI_Datatype recvtype,
                                     MPIL_Topo* topo,
                                     MPIL_Comm* comm,
                                     MPIL_Info* info,
                                     MPIL_Request** request_ptr)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    MPIL_Request* request;
    init_request(&request);
    
    request->start_function = alltoallv_rma_start;
    request->wait_function = alltoallv_rma_wait;

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int tag;
    MPIL_Comm_tag(comm, &tag);

    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;
    request->n_puts = topo->outdegree;
    request->sdispls = (int*)malloc(topo->outdegree*sizeof(int));
    request->put_displs = (int*)malloc(topo->outdegree*sizeof(int));
    request->put_bytes = (int*)malloc(topo->outdegree*sizeof(int));

    int bytes = 0;
    for (int i = 0; i < topo->outdegree; i++)
        bytes += (recvcounts[i] * recv_bytes);
    MPIL_Request_win_init(request, recvbuf, bytes, 1, comm->global_comm);

    for (int i = 0; i < topo->outdegree; i++)
    {
        request->sdispls[i] = sdispls[i] * send_bytes;
        request->put_bytes[i] = sendcounts[i] * send_bytes;
    }


    std::vector<MPI_Request> req(topo->outdegree + topo->indegree);
    for (int i = 0; i < topo->outdegree; i++)
    {
        MPI_Irecv(&(request->put_bytes[i]), 1, MPI_INT, topo->destinations[i],
                tag, comm->global_comm, &(req[i]));
    } 
    for (int i = 0; i < topo->indegree; i++)
    {
        MPI_Isend(&(rdispls[i]), 1, MPI_INT, topo->sources[i],
                tag, comm->global_comm, &(req[topo->outdegree+i]));
    }
    if (topo->outdegree + topo->indegree)
    {
        MPI_Waitall(topo->outdegree + topo->indegree,
                req.data(), MPI_STATUSES_IGNORE);
    }
    for (int i = 0; i < topo->outdegree; i++)
    {
        request->put_displs[i] *= recv_bytes;
    }

    *request_ptr = request;
 
    return MPI_SUCCESS;
}

