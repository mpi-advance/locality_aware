#include "neighborhood/neighborhood_init.h"
//#include "persistent/MPIL_Request.h"
#include "persistent/MPIL_Request.h"
#include "communicator/MPIL_Comm.h"


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

    request->global_n_msgs = topo->indegree + topo->outdegree;
    allocate_requests(request->global_n_msgs, &(request->global_requests));

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
                              comm->global_comm,
                              &(request->global_requests[i]));
    }

    for (int i = 0; i < topo->outdegree; i++)
    {
        ierr += MPI_Send_init(&(send_buffer[sdispls[i] * send_size]),
                              sendcounts[i],
                              sendtype,
                              topo->destinations[i],
                              tag,
                              comm->global_comm,
                              &(request->global_requests[topo->indegree + i]));
    }

    *request_ptr = request;

    return ierr;
}
