#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"
#include "neighborhood/MPIL_Topo.h"
#include "neighborhood/neighborhood_init.h"
#include "persistent/MPIL_Request.h"

int neighbor_alltoallv_init_coll_a2a(const void* sendbuf,
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

    int tag;
    MPIL_Comm_tag(comm, &tag);

    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    const char* send_buffer = (const char*)(sendbuf);
    char* recv_buffer       = (char*)(recvbuf);
    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    int proc, ierr;

    std::vector<int> coll_sendcounts(num_procs, 0);
    std::vector<int> coll_sdispls(num_procs+1);
    std::vector<int> coll_recvcounts(num_procs, 0);
    std::vector<int> coll_rdispls(num_procs+1);

    int ssize = 0;
    for (int i = 0; i < topo->outdegree; i++)
    {
        coll_sendcounts[topo->destinations[i]] = sendcounts[i];
    }
    coll_sdispls[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        coll_sdispls[i+1] = coll_sdispls[i] + coll_sendcounts[i];
    }
    char* coll_sendbuf = (char*)malloc(coll_sdispls[num_procs]*send_size);
    int* coll_sindices = (int*)malloc(coll_sdispls[num_procs]*sizeof(int));

    for (int i = 0; i < topo->outdegree; i++)
    {
        proc = topo->destinations[i];
        for (int j = 0; j < coll_sendcounts[proc]; j++)
        {
            coll_sindices[j] = coll_sdispls[proc] + j;
        }
    }

    int rsize = 0;
    for (int i = 0; i < topo->indegree; i++)
    {
        coll_recvcounts[topo->sources[i]] = recvcounts[i];
    }
    coll_rdispls[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        coll_rdispls[i+1] = coll_rdispls[i] + coll_recvcounts[i];
    }
    char* coll_recvbuf = (char*)malloc(coll_rdispls[num_procs]*recv_size);
    int* coll_rindices = (int*)malloc(coll_rdispls[num_procs]*sizeof(int));

    ierr = MPIL_Alltoallv_init(coll_sendbuf, coll_sendcounts.data(), coll_sdispls.data(), sendtype,
            coll_recvbuf, coll_recvcounts.data(), coll_rdispls.data(), recvtype, 
            comm, info, &(request));

    for (int i = 0; i < topo->indegree; i++)
    {
        proc = topo->sources[i];
        for (int j = 0; j < coll_recvcounts[proc]; j++)
        {
            coll_rindices[j] = coll_rdispls[proc] + j;
        }
    }

    init_neighbor_request(&request->local_L_request);

    request->local_L_request->tmp_sendbuf = coll_sendbuf;
    request->local_L_request->tmp_recvbuf = coll_recvbuf;
    request->local_L_request->send_indices = coll_sindices;
    request->local_L_request->recv_indices = coll_rindices;
    request->local_L_request->size_sends = coll_sdispls[num_procs];
    request->local_L_request->size_recvs = coll_rdispls[num_procs];

    *request_ptr = request;

    return ierr;
}
