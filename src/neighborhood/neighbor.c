#include "neighbor.h"
#include "neighbor_persistent.h"

// Topology object based neighbor alltoallv
int MPIX_Neighbor_topo_alltoallv(
        const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Topo* topo,
        MPIX_Comm* comm)
{
    int tag = 349529;

    MPI_Request* send_requests = (MPI_Request*)malloc((*topo).outdegree*sizeof(MPI_Request));
    MPI_Request* recv_requests = (MPI_Request*)malloc((*topo).indegree*sizeof(MPI_Request));

    const char* send_buffer = (char*) sendbuf;
    char* recv_buffer = (char*) recvbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    for (int i = 0; i < (*topo).indegree; i++)
    {
        MPI_Irecv(&(recv_buffer[rdispls[i]*recv_size]),
                recvcounts[i],
                recvtype,
                topo->sources[i],
                tag,
                comm->global_comm,
                &(recv_requests[i]));
    }

    for (int i = 0; i < (*topo).outdegree; i++)
    {
        MPI_Isend(&(send_buffer[sdispls[i]*send_size]),
                sendcounts[i],
                sendtype,
                topo->destinations[i],
                tag,
                comm->global_comm,
                &(send_requests[i]));
    }

    MPI_Waitall(topo->indegree, recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(topo->outdegree, send_requests, MPI_STATUSES_IGNORE);

    free(send_requests);
    free(recv_requests);

    return MPI_SUCCESS;
}


int MPIX_Neighbor_alltoallv(
        const void* sendbuffer,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuffer,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    MPIX_Topo* topo;
    MPIX_Topo_from_neighbor_comm(comm, &topo);

    MPIX_Neighbor_topo_alltoallv(sendbuffer, sendcounts, sdispls, sendtype,
            recvbuffer, recvcounts, rdispls, recvtype, topo, comm);

    MPIX_Topo_free(&topo);

    return MPI_SUCCESS;

}

