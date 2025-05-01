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

    MPI_Request* send_requests = NULL;
    MPI_Request* recv_requests = NULL;

    const char* send_buffer = NULL;
    char* recv_buffer = NULL;

    if (indegree)
    {
        recv_buffer = (char*) recvbuffer;
        sources = (int*)malloc(indegree*sizeof(int));
        sourceweights = (int*)malloc(indegree*sizeof(int));
        recv_requests = (MPI_Request*)malloc(indegree*sizeof(MPI_Request));
    }

    if (outdegree)
    {
        send_buffer = (char*) sendbuffer;
        destinations = (int*)malloc(outdegree*sizeof(int));
        destweights = (int*)malloc(outdegree*sizeof(int));
        send_requests = (MPI_Request*)malloc(outdegree*sizeof(MPI_Request));
    }

    MPI_Dist_graph_neighbors(
            comm->neighbor_comm, 
            indegree, 
            sources, 
            sourceweights,
            outdegree, 
            destinations, 
            destweights);

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
                comm->global_comm, 
                &(recv_requests[i]));
    }

    for (int i = 0; i < outdegree; i++)
    {
        MPI_Isend(&(send_buffer[sdispls[i]*send_size]),
                sendcounts[i],
                sendtype,
                destinations[i],
                tag,
                comm->global_comm,
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


// TODO : terrible implementation if not using persistent
// Fix this to use aggregation similar to dynamic
// Just combine messages in a 2 step approach
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
    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

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

    //MPIX_Start(request);
    //MPIX_Wait(request, &status);
    //MPIX_Request_free(&request);
    MPIX_Info_free(&xinfo);

    return ierr;
}


// Should a non-persistent version of this exist?
// Can we cheaply remove duplicate values in a 2step approach
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

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

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
