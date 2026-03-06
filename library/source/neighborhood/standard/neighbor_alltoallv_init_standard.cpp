#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"
#include "neighborhood/MPIL_Topo.h"
#include "neighborhood/neighborhood_init.h"
#include "persistent/MPIL_Request.h"

int neighbor_alltoallv_init_standard(const void* sendbuf,
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
    MPIL_Request* request;
    init_neighbor_request(&request);

    int tag;
    MPIL_Comm_tag(comm, &tag);

    int ierr = neighbor_alltoallv_init_standard_helper(sendbuf, sendcounts, sdispls,
                    sendtype, recvbuf, recvcounts, rdispls, recvtype, topo, comm->global_comm, 
                    info, tag, request);

    *request_ptr = request;

    return ierr;
}



int neighbor_alltoallv_init_standard_helper(const void* sendbuf,
                                     const int sendcounts[],
                                     const int sdispls[],
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcounts[],
                                     const int rdispls[],
                                     MPI_Datatype recvtype,
                                     MPIL_Topo* topo,
                                     MPI_Comm comm,
                                     MPIL_Info* info,
                                     int tag,
                                     MPIL_Request* request)
{
    allocate_requests(topo->indegree + topo->outdegree, request);

    const char* send_buffer = (const char*)(sendbuf);
    char* recv_buffer       = (char*)(recvbuf);
    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    int ierr = 0;

    for (int i = 0; i < topo->indegree; i++)
    {
        ierr += MPI_Recv_init(&(recv_buffer[rdispls[i] * recv_size]),
                              recvcounts[i],
                              recvtype,
                              topo->sources[i],
                              tag,
                              comm,
                              &(request->requests[i]));
    }

    for (int i = 0; i < topo->outdegree; i++)
    {
        ierr += MPI_Send_init(&(send_buffer[sdispls[i] * send_size]),
                              sendcounts[i],
                              sendtype,
                              topo->destinations[i],
                              tag,
                              comm,
                              &(request->requests[topo->indegree + i]));
    }

    return ierr;
}
