#include "neighbor.h"

void init_request(MPIX_Request** request_ptr)
{
    MPIX_Request* request = (MPIX_Request*)malloc(sizeof(MPIX_Request));

    request->locality = NULL;

    request->local_L_n_msgs = 0;
    request->local_S_n_msgs = 0;
    request->local_R_n_msgs = 0;
    request->global_n_msgs = 0;

    request->local_L_requests = NULL;
    request->local_S_requests = NULL;
    request->local_R_requests = NULL;
    request->global_requests = NULL;

    request->recv_size = 0;

    *request_ptr = request;
}

void destroy_request(MPIX_Request* request)
{
    if (request->local_L_n_msgs)
        free(request->local_L_requests);
    if (request->local_S_n_msgs)
        free(request->local_S_requests);
    if (request->local_R_n_msgs)
        free(request->local_R_requests);
    if (request->global_n_msgs)
        free(request->global_requests);

    if (request->locality)
        destroy_locality_comm(request->locality);

    free(request);
}

void allocate_requests(int n_requests, MPI_Request** request_ptr)
{
    if (n_requests)
    {
        MPI_Request* request = (MPI_Request*)malloc(sizeof(MPI_Request)*n_requests);
        *request_ptr = request;
    }
    else *request_ptr = NULL;
}

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
        MPIX_Comm* comm)
{

    MPIX_Request* request;
    MPI_Status status;

    int ierr = MPIX_Neighbor_alltoallv_init(sendbuffer,
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

//    MPIX_Start(request);
//    MPIX_Wait(request, &status);
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
    init_request(&request);

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
                comm->neighbor_comm, 
                &(request->global_requests[i]));
    }

    for (int i = 0; i < outdegree; i++)
    {
        MPI_Send_init(&(send_buffer[sdispls[i]*send_size]),
                sendcounts[i],
                sendtype,
                destinations[i],
                tag,
                comm->neighbor_comm,
                &(request->global_requests[indegree+i]));
    }

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
    init_request(&request);

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

    *request_ptr = request;

    return MPI_SUCCESS;
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

    int tag = 304591;
    int indegree, outdegree, weighted;
    MPI_Dist_graph_neighbors_count(
            comm->neighbor_comm, 
            &indegree, 
            &outdegree, 
            &weighted);

    int* sources = (int*)malloc(indegree*sizeof(int));
    int* sourceweights = (int*)malloc(indegree*sizeof(int));
    int* destinations = (int*)malloc(outdegree*sizeof(int));
    int* destweights = (int*)malloc(outdegree*sizeof(int));
    MPI_Dist_graph_neighbors(
            comm->neighbor_comm, 
            indegree, 
            sources, 
            sourceweights,
            outdegree, 
            destinations, 
            destweights);

    MPIX_Request* request;
    init_request(&request);

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

    *request_ptr = request;

    return 0;

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

    int tag = 304591;
    int indegree, outdegree, weighted;
    MPI_Dist_graph_neighbors_count(
            comm->neighbor_comm, 
            &indegree, 
            &outdegree, 
            &weighted);

    int* sources = (int*)malloc(indegree*sizeof(int));
    int* sourceweights = (int*)malloc(indegree*sizeof(int));
    int* destinations = (int*)malloc(outdegree*sizeof(int));
    int* destweights = (int*)malloc(outdegree*sizeof(int));
    MPI_Dist_graph_neighbors(
            comm->neighbor_comm, 
            indegree, 
            sources, 
            sourceweights,
            outdegree, 
            destinations, 
            destweights);

    long send_size = 0;
    for (int i = 0; i < outdegree; i++)
        send_size += sendcounts[i];
    long recv_size = 0;
    for (int i = 0; i < indegree; i++)
        recv_size += recvcounts[i];
    long first_send;
    MPI_Exscan(&send_size, &first_send, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) first_send = 0;

    long* global_send_indices = (long*)malloc(send_size*sizeof(long));
    long* global_recv_indices = (long*)malloc(recv_size*sizeof(long));
    for (int i = 0; i < send_size; i++)
        global_send_indices[i] = first_send + i;

    MPIX_Neighbor_alltoallv(global_send_indices, sendcounts, sdispls, MPI_LONG, 
            global_recv_indices, recvcounts, rdispls, MPI_LONG, comm);

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
