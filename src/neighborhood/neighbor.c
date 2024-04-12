#include "neighbor.h"

int init_communication(const void* sendbuffer,
        int n_sends,
        const int* send_procs,
        const int* send_ptr, 
        MPI_Datatype sendtype,
        void* recvbuffer, 
        int n_recvs,
        const int* recv_procs,
        const int* recv_ptr,
        MPI_Datatype recvtype,
        int tag,
        MPI_Comm comm,
        int* n_request_ptr,
        MPI_Request** request_ptr)
{
    int ierr, start, size;
    int send_size, recv_size;

    char* send_buffer = (char*) sendbuffer;
    char* recv_buffer = (char*) recvbuffer;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    MPI_Request* requests;
    *n_request_ptr = n_recvs+n_sends;
    allocate_requests(*n_request_ptr, &requests);

    for (int i = 0; i < n_recvs; i++)
    {
        start = recv_ptr[i];
        size = recv_ptr[i+1] - start;

        ierr += MPI_Recv_init(&(recv_buffer[start*recv_size]), 
                size, 
                recvtype, 
                recv_procs[i],
                tag,
                comm, 
                &(requests[i]));
    }

    for (int i = 0; i < n_sends; i++)
    {
        start = send_ptr[i];
        size = send_ptr[i+1] - start;

        ierr += MPI_Send_init(&(send_buffer[start*send_size]),
                size,
                sendtype,
                send_procs[i],
                tag,
                comm,
                &(requests[n_recvs+i]));
    }

    *request_ptr = requests;

    return ierr;
}


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
            MPI_INFO_NULL,
            &request);

    MPIX_Start(request);
    MPIX_Wait(request, &status);
    MPIX_Request_free(request);

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
        MPI_Comm comm)
{

    int tag = 349526;

    int indegree, outdegree, weighted;
    MPI_Dist_graph_neighbors_count(
            comm, 
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
            comm, 
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
                comm, 
                &(recv_requests[i]));
    }

    for (int i = 0; i < outdegree; i++)
    {
        MPI_Isend(&(send_buffer[sdispls[i]*send_size]),
                sendcounts[i],
                sendtype,
                destinations[i],
                tag,
                comm,
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

int MPIX_Neighbor_topo_alltoallv_init(
        const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Topo* topo,
        MPI_Comm comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
    int tag = 349526;

    int indegree, outdegree, weighted;
    MPIX_Topo_dist_graph_neighbors_count(
            topo, 
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

    MPIX_Topo_dist_graph_neighbors(
            topo, 
            indegree, 
            sources, 
            sourceweights,
            outdegree, 
            destinations, 
            destweights); 

    MPIX_Request* request;
    MPIX_Request_init(&request);

    request->global_n_msgs = indegree+outdegree;
    allocate_requests(request->global_n_msgs, &(request->global_requests));

    const char* send_buffer = (char*) sendbuf;
    char* recv_buffer = (char*) recvbuf;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    for (int i = 0; i < indegree; i++)
    {
        MPI_Recv_init(&(recv_buffer[rdispls[i]*recv_size]), 
                recvcounts[i],
                recvtype, 
                sources[i],
                tag,
                comm, 
                &(request->global_requests[i]));
    }

    for (int i = 0; i < outdegree; i++)
    {
        MPI_Send_init(&(send_buffer[sdispls[i]*send_size]),
                sendcounts[i],
                sendtype,
                destinations[i],
                tag,
                comm,
                &(request->global_requests[indegree+i]));
    }

    free(sources);
    free(sourceweights);
    free(destinations);
    free(destweights);

    request->start_function = neighbor_start;
    request->wait_function = neighbor_wait;

    *request_ptr = request;

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

    int ierr = MPIX_Neighbor_part_locality_alltoallv_init(sendbuffer,
            sendcounts,
            sdispls,
            sendtype,
            recvbuffer,
            recvcounts,
            rdispls,
            recvtype,
            comm,
            MPI_INFO_NULL, 
            &request);

    MPIX_Start(request);
    MPIX_Wait(request, &status);
    MPIX_Request_free(request);

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
            MPI_INFO_NULL, 
            &request);

    MPIX_Start(request);
    MPIX_Wait(request, &status);
    MPIX_Request_free(request);

    return ierr;
}

// Standard Persistent Neighbor Alltoallv
// Extension takes array of requests instead of single request
// 'requests' must be of size indegree+outdegree!
int MPIX_Neighbor_alltoallv_init(
        const void* sendbuffer,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuffer,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
    int tag = 349526;

    int indegree, outdegree, weighted;
    MPI_Dist_graph_neighbors_count(
            comm, 
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
            comm, 
            indegree, 
            sources, 
            sourceweights,
            outdegree, 
            destinations, 
            destweights);

    MPIX_Request* request;
    MPIX_Request_init(&request);
    

    request->global_n_msgs = indegree+outdegree;
    allocate_requests(request->global_n_msgs, &(request->global_requests));

    const char* send_buffer = (char*) sendbuffer;
    char* recv_buffer = (char*) recvbuffer;

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    for (int i = 0; i < indegree; i++)
    {
        MPI_Recv_init(&(recv_buffer[rdispls[i]*recv_size]), 
                recvcounts[i],
                recvtype, 
                sources[i],
                tag,
                comm, 
                &(request->global_requests[i]));
    }

    for (int i = 0; i < outdegree; i++)
    {
        MPI_Send_init(&(send_buffer[sdispls[i]*send_size]),
                sendcounts[i],
                sendtype,
                destinations[i],
                tag,
                comm,
                &(request->global_requests[indegree+i]));
    }

    free(sources);
    free(sourceweights);
    free(destinations);
    free(destweights);

    request->start_function = neighbor_start;
    request->wait_function = neighbor_wait;

    *request_ptr = request;

    return MPI_SUCCESS;
}



// Standard Persistent Neighbor Alltoallv
// Extension takes array of requests instead of single request
// 'requests' must be of size indegree+outdegree!
int MPIX_Neighbor_alltoallw_init(
        const void* sendbuffer,
        const int sendcounts[],
        const MPI_Aint sdispls[],
        MPI_Datatype* sendtypes,
        void* recvbuffer,
        const int recvcounts[],
        const MPI_Aint rdispls[],
        MPI_Datatype* recvtypes,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
    int tag = 349526;

    int indegree, outdegree, weighted;
    MPI_Dist_graph_neighbors_count(
            comm->neighbor_comm, 
            &indegree, 
            &outdegree, 
            &weighted);

    int sources[indegree];
    int sourceweights[indegree];
    int destinations[outdegree];
    int destweights[outdegree];
    MPI_Dist_graph_neighbors(
            comm->neighbor_comm, 
            indegree, 
            sources, 
            sourceweights,
            outdegree, 
            destinations, 
            destweights);

    MPIX_Request* request;
    MPIX_Request_init(&request);

    request->global_n_msgs = indegree+outdegree;
    allocate_requests(request->global_n_msgs, &(request->global_requests));

    const char* send_buffer = (const char*)(sendbuffer);
    char* recv_buffer = (char*)(recvbuffer);

    for (int i = 0; i < outdegree; i++)
    {
        MPI_Send_init(&(send_buffer[sdispls[i]]),
                sendcounts[i],
                sendtypes[i],
                destinations[i],
                tag,
                comm->neighbor_comm,
                &(request->global_requests[indegree+i]));

    }
    for (int i = 0; i < indegree; i++)
    {
        MPI_Recv_init(&(recv_buffer[rdispls[i]]),
                recvcounts[i], 
                recvtypes[i], 
                sources[i],
                tag,
                comm->neighbor_comm, 
                &(request->global_requests[i]));

    }

    request->start_function = neighbor_start;
    request->wait_function = neighbor_wait;

    *request_ptr = request;

    return MPI_SUCCESS;
}

int MPIX_Neighbor_locality_topo_alltoallv_init(
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
        MPIX_Topo* topo,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
    int tag = 304591;
    int indegree, outdegree, weighted;
    MPIX_Topo_dist_graph_neighbors_count(
            topo, 
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

    MPIX_Topo_dist_graph_neighbors(
            topo, 
            indegree, 
            sources, 
            sourceweights,
            outdegree, 
            destinations, 
            destweights);

    MPIX_Request* request;
    MPIX_Request_init(&request);

    // Initialize Locality-Aware Communication Strategy (3-Step)
    // E.G. Determine which processes talk to eachother at every step
    // TODO : instead of mpi_comm, use comm
    //        - will need to create local_comm in dist_graph_create_adjacent...
    init_locality(outdegree,
            destinations,
            sdispls,
            sendcounts,
            indegree,
            sources,
            rdispls,
            recvcounts,
            global_sindices,
            global_rindices,
            sendtype,
            recvtype,
            comm, // communicator used in dist_graph_create_adjacent
            request);

    request->sendbuf = sendbuffer;
    request->recvbuf = recvbuffer;
    MPI_Type_size(recvtype, &(request->recv_size));

    // Local L Communication
    init_communication(request->locality->local_L_comm->send_data->buffer,
            request->locality->local_L_comm->send_data->num_msgs,
            request->locality->local_L_comm->send_data->procs,
            request->locality->local_L_comm->send_data->indptr,
            sendtype,
            request->locality->local_L_comm->recv_data->buffer,
            request->locality->local_L_comm->recv_data->num_msgs,
            request->locality->local_L_comm->recv_data->procs,
            request->locality->local_L_comm->recv_data->indptr,
            recvtype,
            request->locality->local_L_comm->tag,
            comm->local_comm,
            &(request->local_L_n_msgs),
            &(request->local_L_requests));

    // Local S Communication
    init_communication(request->locality->local_S_comm->send_data->buffer,
            request->locality->local_S_comm->send_data->num_msgs,
            request->locality->local_S_comm->send_data->procs,
            request->locality->local_S_comm->send_data->indptr,
            sendtype,
            request->locality->local_S_comm->recv_data->buffer,
            request->locality->local_S_comm->recv_data->num_msgs,
            request->locality->local_S_comm->recv_data->procs,
            request->locality->local_S_comm->recv_data->indptr,
            recvtype,
            request->locality->local_S_comm->tag,
            comm->local_comm,
            &(request->local_S_n_msgs),
            &(request->local_S_requests));

    // Global Communication
    init_communication(request->locality->global_comm->send_data->buffer,
            request->locality->global_comm->send_data->num_msgs,
            request->locality->global_comm->send_data->procs,
            request->locality->global_comm->send_data->indptr,
            sendtype,
            request->locality->global_comm->recv_data->buffer,
            request->locality->global_comm->recv_data->num_msgs,
            request->locality->global_comm->recv_data->procs,
            request->locality->global_comm->recv_data->indptr,
            recvtype,
            request->locality->global_comm->tag,
            comm->global_comm,
            &(request->global_n_msgs),
            &(request->global_requests));

    // Local R Communication
    init_communication(request->locality->local_R_comm->send_data->buffer,
            request->locality->local_R_comm->send_data->num_msgs,
            request->locality->local_R_comm->send_data->procs,
            request->locality->local_R_comm->send_data->indptr,
            sendtype,
            request->locality->local_R_comm->recv_data->buffer,
            request->locality->local_R_comm->recv_data->num_msgs,
            request->locality->local_R_comm->recv_data->procs,
            request->locality->local_R_comm->recv_data->indptr,
            recvtype,
            request->locality->local_R_comm->tag,
            comm->local_comm,
            &(request->local_R_n_msgs),
            &(request->local_R_requests));

    free(sources);
    free(sourceweights);
    free(destinations);
    free(destweights);

    request->start_function = neighbor_start;
    request->wait_function = neighbor_wait;

    *request_ptr = request;

    return 0;

}

// Locality-Aware Extension to Persistent Neighbor Alltoallv
// Needs global indices for each send and receive
int MPIX_Neighbor_locality_alltoallv_init(
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
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
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

    MPIX_Request* request;
    MPIX_Request_init(&request);

    // Initialize Locality-Aware Communication Strategy (3-Step)
    // E.G. Determine which processes talk to eachother at every step
    // TODO : instead of mpi_comm, use comm
    //        - will need to create local_comm in dist_graph_create_adjacent...
    init_locality(outdegree, 
            destinations, 
            sdispls, 
            sendcounts,
            indegree, 
            sources, 
            rdispls,
            recvcounts,
            global_sindices,
            global_rindices,
            sendtype,
            recvtype,
            comm, // communicator used in dist_graph_create_adjacent 
            request);

    request->sendbuf = sendbuffer;
    request->recvbuf = recvbuffer;
    MPI_Type_size(recvtype, &(request->recv_size));

    // Local L Communication
    init_communication(request->locality->local_L_comm->send_data->buffer,
            request->locality->local_L_comm->send_data->num_msgs,
            request->locality->local_L_comm->send_data->procs,
            request->locality->local_L_comm->send_data->indptr,
            sendtype,
            request->locality->local_L_comm->recv_data->buffer,
            request->locality->local_L_comm->recv_data->num_msgs,
            request->locality->local_L_comm->recv_data->procs,
            request->locality->local_L_comm->recv_data->indptr,
            recvtype,
            request->locality->local_L_comm->tag,
            comm->local_comm,
            &(request->local_L_n_msgs),
            &(request->local_L_requests));

    // Local S Communication
    init_communication(request->locality->local_S_comm->send_data->buffer,
            request->locality->local_S_comm->send_data->num_msgs,
            request->locality->local_S_comm->send_data->procs,
            request->locality->local_S_comm->send_data->indptr,
            sendtype,
            request->locality->local_S_comm->recv_data->buffer,
            request->locality->local_S_comm->recv_data->num_msgs,
            request->locality->local_S_comm->recv_data->procs,
            request->locality->local_S_comm->recv_data->indptr,
            recvtype,
            request->locality->local_S_comm->tag,
            comm->local_comm,
            &(request->local_S_n_msgs),
            &(request->local_S_requests));

    // Global Communication
    init_communication(request->locality->global_comm->send_data->buffer,
            request->locality->global_comm->send_data->num_msgs,
            request->locality->global_comm->send_data->procs,
            request->locality->global_comm->send_data->indptr,
            sendtype,
            request->locality->global_comm->recv_data->buffer,
            request->locality->global_comm->recv_data->num_msgs,
            request->locality->global_comm->recv_data->procs,
            request->locality->global_comm->recv_data->indptr,
            recvtype,
            request->locality->global_comm->tag,
            comm->global_comm,
            &(request->global_n_msgs),
            &(request->global_requests));

    // Local R Communication
    init_communication(request->locality->local_R_comm->send_data->buffer,
            request->locality->local_R_comm->send_data->num_msgs,
            request->locality->local_R_comm->send_data->procs,
            request->locality->local_R_comm->send_data->indptr,
            sendtype,
            request->locality->local_R_comm->recv_data->buffer,
            request->locality->local_R_comm->recv_data->num_msgs,
            request->locality->local_R_comm->recv_data->procs,
            request->locality->local_R_comm->recv_data->indptr,
            recvtype,
            request->locality->local_R_comm->tag,
            comm->local_comm,
            &(request->local_R_n_msgs),
            &(request->local_R_requests));

    free(sources);
    free(sourceweights);
    free(destinations);
    free(destweights);

    request->start_function = neighbor_start;
    request->wait_function = neighbor_wait;

    *request_ptr = request;

    return 0;

}

int MPIX_Neighbor_part_locality_topo_alltoallv_init(
        const void* sendbuffer,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuffer,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Topo* topo,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
    int rank;
    MPI_Comm_rank(comm->global_comm, &rank);

    int tag = 304591;
    int indegree, outdegree, weighted;
    MPIX_Topo_dist_graph_neighbors_count(
            topo, 
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

    MPIX_Topo_dist_graph_neighbors(
            topo, 
            indegree, 
            sources, 
            sourceweights,
            outdegree, 
            destinations, 
            destweights);

    // Make separate temporary displs incase sendbuffer/recvbuffer are not contiguous
    int* send_displs = (int*)malloc(outdegree*sizeof(int));
    int* recv_displs = (int*)malloc(indegree*sizeof(int));

    long send_size = 0;
    for (int i = 0; i < outdegree; i++)
    {
        send_displs[i] = send_size;
        send_size += sendcounts[i];
    }
    long recv_size = 0;
    for (int i = 0; i < indegree; i++)
    {
        recv_displs[i] = recv_size;
        recv_size += recvcounts[i];
    }

    long first_send;
    MPI_Exscan(&send_size, &first_send, 1, MPI_LONG, MPI_SUM, comm->global_comm);
    if (rank == 0) first_send = 0;

    long* global_send_indices = NULL;
    long* global_recv_indices = NULL;

    if (send_size)
        global_send_indices = (long*)malloc(send_size*sizeof(long));
    if (recv_size)
        global_recv_indices = (long*)malloc(recv_size*sizeof(long));
    for (int i = 0; i < send_size; i++)
        global_send_indices[i] = first_send + i;


    MPIX_Neighbor_topo_alltoallv(global_send_indices, sendcounts, send_displs, MPI_LONG,
            global_recv_indices, recvcounts, recv_displs, MPI_LONG, topo, comm->global_comm);

    free(send_displs);
    free(recv_displs);

    free(sources);
    free(sourceweights);
    free(destinations);
    free(destweights);

    int err = MPIX_Neighbor_locality_topo_alltoallv_init(sendbuffer, sendcounts, sdispls,
            global_send_indices, sendtype, recvbuffer, recvcounts, rdispls,
            global_recv_indices, recvtype, topo, comm, info, request_ptr);

    free(global_send_indices);
    free(global_recv_indices);

    return err;
}

int MPIX_Neighbor_part_locality_alltoallv_init(
        const void* sendbuffer,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuffer,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
    int rank; 
    MPI_Comm_rank(comm->global_comm, &rank);

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

    // Make separate temporary displs incase sendbuffer/recvbuffer are not contiguous
    int* send_displs = (int*)malloc(outdegree*sizeof(int));
    int* recv_displs = (int*)malloc(indegree*sizeof(int));

    long send_size = 0;
    for (int i = 0; i < outdegree; i++)
    {
        send_displs[i] = send_size;
        send_size += sendcounts[i];
    }
    long recv_size = 0;
    for (int i = 0; i < indegree; i++)
    {
        recv_displs[i] = recv_size;
        recv_size += recvcounts[i];
    }

    long first_send;
    MPI_Exscan(&send_size, &first_send, 1, MPI_LONG, MPI_SUM, comm->global_comm);
    if (rank == 0) first_send = 0;

    long* global_send_indices = NULL;
    long* global_recv_indices = NULL;

    if (send_size)
        global_send_indices = (long*)malloc(send_size*sizeof(long));
    if (recv_size)
        global_recv_indices = (long*)malloc(recv_size*sizeof(long));
    for (int i = 0; i < send_size; i++)
        global_send_indices[i] = first_send + i;


    MPIX_Neighbor_alltoallv(global_send_indices, sendcounts, send_displs, MPI_LONG, 
            global_recv_indices, recvcounts, recv_displs, MPI_LONG, comm->neighbor_comm);

    free(send_displs);
    free(recv_displs);

    free(sources);
    free(sourceweights);
    free(destinations);
    free(destweights);

    int err = MPIX_Neighbor_locality_alltoallv_init(sendbuffer, sendcounts, sdispls, 
            global_send_indices, sendtype, recvbuffer, recvcounts, rdispls, 
            global_recv_indices, recvtype, comm, info, request_ptr);

    free(global_send_indices);
    free(global_recv_indices);

    return err;
}
