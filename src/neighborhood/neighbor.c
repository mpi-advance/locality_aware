#include "neighbor.h"
#include "neighbor_persistent.h"

int MPIX_Neighbor_alltoallw(
        const void* sendbuf,
        const int sendcounts[],
        const MPI_Aint sdispls[],
        MPI_Datatype* sendtypes,
        void* recvbuf,
        const int recvcounts[],
        const MPI_Aint rdispls[],
        MPI_Datatype* recvtypes,
        MPIX_Comm* comm)
{

    MPIX_Request* request;
    MPI_Status status;

    MPIX_Info* xinfo;
    MPIX_Info_init(&xinfo);

    int ierr = MPIX_Neighbor_alltoallw_init(
            sendbuf,
            sendcounts,
            sdispls,
            sendtypes,
            recvbuf,
            recvcounts,
            rdispls,
            recvtypes,
            comm,
            xinfo,
            &request);

    MPIX_Start(request);
    MPIX_Wait(request, &status);
    MPIX_Request_free(&request);

    MPIX_Info_free(&xinfo);

    return ierr;
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

    int tag = 349526;

    int indegree, outdegree, weighted;
    MPI_Dist_graph_neighbors_count(
            comm->neighbor_comm, 
            &indegree, 
            &outdegree, 
            &weighted);

    int* sources = NULL;
    int* sourceweights = NULL;
    int* destinations = NULL;
    int* destweights = NULL;

    if (indegree)
    {
        sources = (int*)malloc(indegree*sizeof(int));
        sourceweights = (int*)malloc(indegree*sizeof(int));
    }

    if (outdegree)
    {
        destinations = (int*)malloc(outdegree*sizeof(int));
        destweights = (int*)malloc(outdegree*sizeof(int));
    }

    MPI_Dist_graph_neighbors(
            comm->neighbor_comm, 
            indegree, 
            sources, 
            sourceweights,
            outdegree, 
            destinations, 
            destweights);

    MPI_Request* send_requests = (MPI_Request*)malloc(outdegree*sizeof(MPI_Request));
    MPI_Request* recv_requests = (MPI_Request*)malloc(indegree*sizeof(MPI_Request));

    const char* send_buffer = (char*) sendbuffer;
    char* recv_buffer = (char*) recvbuffer;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    for (int i = 0; i < indegree; i++)
    {
        MPI_Irecv(&(recv_buffer[rdispls[i]*recv_size]), 
                recvcounts[i],
                recvtype, 
                sources[i],
                tag,
                comm->neighbor_comm, 
                &(recv_requests[i]));
    }

    for (int i = 0; i < outdegree; i++)
    {
        MPI_Isend(&(send_buffer[sdispls[i]*send_size]),
                sendcounts[i],
                sendtype,
                destinations[i],
                tag,
                comm->neighbor_comm,
                &(send_requests[i]));
    }

    MPI_Waitall(indegree, recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(outdegree, send_requests, MPI_STATUSES_IGNORE);

    free(sources);
    free(sourceweights);
    free(destinations);
    free(destweights);

    free(send_requests);
    free(recv_requests);

    return MPI_SUCCESS;

}


int MPIX_Neighbor_part_locality_alltoallv(
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

    MPIX_Request* request;
    MPI_Status status;

    MPIX_Info* xinfo;
    MPIX_Info_init(&xinfo);

    int ierr = MPIX_Neighbor_part_locality_alltoallv_init(sendbuffer,
            sendcounts,
            sdispls,
            sendtype,
            recvbuffer,
            recvcounts,
            rdispls,
            recvtype,
            comm,
            xinfo,
            &request);

    MPIX_Start(request);
    MPIX_Wait(request, &status);
    MPIX_Request_free(&request);
    MPIX_Info_free(&xinfo);

    return ierr;
}

int MPIX_Neighbor_locality_alltoallv(
        const void* sendbuffer,
        const int sendcounts[],
        const int sdispls[],
        const long global_sindices[],
        MPI_Datatype sendtype,
        void* recvbuffer,
        const int recvcounts[],
        const int rdispls[],
        const long global_rindices[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{

    MPIX_Request* request;
    MPI_Status status;
    MPIX_Info* xinfo;
    MPIX_Info_init(&xinfo);

    int ierr = MPIX_Neighbor_locality_alltoallv_init(sendbuffer,
            sendcounts,
            sdispls,
            global_sindices,
            sendtype,
            recvbuffer,
            recvcounts,
            rdispls,
            global_rindices,
            recvtype,
            comm,
            xinfo,
            &request);

    MPIX_Start(request);
    MPIX_Wait(request, &status);
    MPIX_Request_free(&request);
    MPIX_Info_free(&xinfo);

    return ierr;
}


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
        MPI_Comm comm)
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
                (*topo).sources[i],
                tag,
                comm,
                &(recv_requests[i]));
    }

    for (int i = 0; i < (*topo).outdegree; i++)
    {
        MPI_Isend(&(send_buffer[sdispls[i]*send_size]),
                sendcounts[i],
                sendtype,
                (*topo).destinations[i],
                tag,
                comm,
                &(send_requests[i]));
    }

    MPI_Waitall((*topo).indegree, recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall((*topo).outdegree, send_requests, MPI_STATUSES_IGNORE);

    free(send_requests);
    free(recv_requests);

    return MPI_SUCCESS;
}

