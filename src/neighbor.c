#include "neighbor.h"
#include "locality_comm.h"

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

int init_communication(const void* sendbuf,
        int n_sends,
        const int* send_procs,
        const int* send_ptr, 
        MPI_Datatype sendtype,
        void* recvbuf, 
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

    char* send_buffer = (char*) sendbuf;
    char* recv_buffer = (char*) recvbuf;
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

        ierr += MPI_Send_init(&(sendbuf[start*send_size]),
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


// Standard Persistent Neighbor Alltoallv
// Extension takes array of requests instead of single request
// 'requests' must be of size indegree+outdegree!
int MPIX_Neighbor_alltoallv_init(
        const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPIX_Comm* comm,
        MPI_Info info,
        MPIX_Request** request_ptr)
{
    int ierr = 0;
    int tag = 349526;

    int indegree, outdegree, weighted;
    ierr += MPI_Dist_graph_neighbors_count(
            comm->neighbor_comm, 
            &indegree, 
            &outdegree, 
            &weighted);

    int sources[indegree];
    int sourceweights[indegree];
    int destinations[outdegree];
    int destweights[outdegree];
    ierr += MPI_Dist_graph_neighbors(
            comm->neighbor_comm, 
            indegree, 
            sources, 
            sourceweights,
            outdegree, 
            destinations, 
            destweights);

    MPIX_Request* request;
    init_request(&request);

    init_communication(
            sendbuf, 
            outdegree, 
            destinations,
            sdispls,
            sendtype,
            recvbuf,
            indegree,
            sources,
            rdispls,
            recvtype,
            tag,
            comm->neighbor_comm,
            &(request->global_n_msgs),
            &(request->global_requests));

    *request_ptr = request;

    return ierr;
}


/*
// Locality-Aware Extension to Persistent Neighbor Alltoallv
// Needs global indices for each send and receive
int MPIX_Neighbor_alltoallv_init(
        const void* sendbuf,
        const int sendcounts[],
        const int sdispls[],
        const int global_sindices[],
        MPI_Datatype sendtype,
        void* recvbuf,
        const int recvcounts[],
        const int rdispls[],
        const int global_rindices[],
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

    // Initialize Locality-Aware Communication Strategy (3-Step)
    // E.G. Determine which processes talk to eachother at every step
    // TODO : instead of mpi_comm, use comm
    //        - will need to create local_comm in dist_graph_create_adjacent...
    init_locality(outdegree, 
            destinations, 
            sdispls, 
            dest_indices,
            indegree, 
            sources, 
            rdispls,
            global_dest_indices,
            global_source_indices,
            comm->global_comm, // communicator used in dist_graph_create_adjacent 
            request);

    // Local L Communication
    init_communication(sendbuf,
            request->nap_comm->local_L_comm->send_data->num_msgs,
            request->nap_comm->local_L_comm->send_data->procs,
            request->nap_comm->local_L_comm->send_data->indptr,
            sendtype,
            request->nap_comm->local_L_comm->recv_data->buf,
            request->nap_comm->local_L_comm->recv_data->num_msgs,
            request->nap_comm->local_L_comm->recv_data->procs,
            request->nap_comm->local_L_comm->recv_data->indptr,
            recvtype,
            request->nap_comm->local_L_comm->tag,
            comm->local_comm,
            &(request->local_L_n_msgs),
            &(request->local_L_requests));

    // Local S Communication
    init_communication(sendbuf,
            request->nap_comm->local_S_comm->send_data->num_msgs,
            request->nap_comm->local_S_comm->send_data->procs,
            request->nap_comm->local_S_comm->send_data->indptr,
            sendtype,
            request->nap_comm->local_S_comm->recv_data->buf,
            request->nap_comm->local_S_comm->recv_data->num_msgs,
            request->nap_comm->local_S_comm->recv_data->procs,
            request->nap_comm->local_S_comm->recv_data->indptr,
            recvtype,
            request->nap_comm->local_S_comm->tag,
            comm->local_comm,
            &(request->local_S_n_msgs),
            &(request->local_S_requests));

    // Global Communication
    init_communication(request->nap_comm->global_comm->send_data->buf,
            request->nap_comm->global_comm->send_data->num_msgs,
            request->nap_comm->global_comm->send_data->procs,
            request->nap_comm->global_comm->send_data->indptr,
            sendtype,
            request->nap_comm->global_comm->recv_data->buf,
            request->nap_comm->global_comm->recv_data->num_msgs,
            request->nap_comm->global_comm->recv_data->procs,
            request->nap_comm->global_comm->recv_data->indptr,
            recvtype,
            request->nap_comm->global_comm->tag,
            comm->global_comm,
            &(request->global_n_msgs),
            &(request->global_requests));


    // Local R Communication
    init_communication(request->nap_comm->local_R_comm->send_data->buf,
            request->nap_comm->local_R_comm->send_data->num_msgs,
            request->nap_comm->local_R_comm->send_data->procs,
            request->nap_comm->local_R_comm->send_data->indptr,
            sendtype,
            request->nap_comm->local_R_comm->recv_data->buf,
            request->nap_comm->local_R_comm->recv_data->num_msgs,
            request->nap_comm->local_R_comm->recv_data->procs,
            request->nap_comm->local_R_comm->recv_data->indptr,
            recvtype,
            request->nap_comm->local_R_comm->tag,
            comm->local_comm,
            &(request->local_R_n_msgs),
            &(request->local_R_requests));

    // Global Communication
    init_communication(request->nap_comm->global_comm->send_data->buf,
            request->nap_comm->global_comm->send_data->num_msgs,
            request->nap_comm->global_comm->send_data->procs,
            request->nap_comm->global_comm->send_data->indptr,
            sendtype,
            request->nap_comm->global_comm->recv_data->buf,
            request->nap_comm->global_comm->recv_data->num_msgs,
            request->nap_comm->global_comm->recv_data->procs,
            request->nap_comm->global_comm->recv_data->indptr,
            recvtype,
            request->nap_comm->global_comm->tag,
            comm->global_comm,
            &(request->global_n_msgs),
            &(request->global_requests));

    *request_ptr = request;

    return 0;

}
    */


// Starting locality-aware requests
// 1. Start Local_L
// 2. Start and wait for local_S
// 3. Start global
int MPIX_Start(MPIX_Request* request)
{
    int ierr;

    // Local L sends sendbuf
    if (request->local_L_n_msgs)
    {
        ierr = MPI_Startall(request->local_L_n_msgs, request->local_L_requests);
    }


    // Local S sends sendbuf
    if (request->local_S_n_msgs)
    {
        ierr = MPI_Startall(request->local_S_n_msgs, request->local_S_requests);
        ierr = MPI_Waitall(request->local_S_n_msgs, request->local_S_requests, MPI_STATUSES_IGNORE);

        // Copy into global->send_data->buffer
    }

    // Global sends buffer in locality, sendbuf in standard
    if (request->global_n_msgs)
        ierr = MPI_Startall(request->global_n_msgs, request->global_requests);

    return ierr;
}


// Wait for locality-aware requests
// 1. Wait for global
// 2. Start and wait for local_R
// 3. Wait for local_L
// TODO : Currently ignores the status!
int MPIX_Wait(MPIX_Request* request, MPI_Status status)
{
    int ierr;

    // Global waits for recvs
    if (request->global_n_msgs)
    {
        ierr = MPI_Waitall(request->global_n_msgs, request->global_requests, MPI_STATUSES_IGNORE);

        // Copy into local_R->send_data->buffer
    }

    // Wait for local_R recvs
    if (request->local_R_n_msgs)
    {
        ierr = MPI_Startall(request->local_R_n_msgs, request->local_R_requests);
        ierr = MPI_Waitall(request->local_R_n_msgs, request->local_R_requests, MPI_STATUSES_IGNORE);

        // Copy into recvbuf
    }

    // Wait for local_L recvs
    if (request->local_L_n_msgs)
    {
        ierr = MPI_Waitall(request->local_L_n_msgs, request->local_L_requests, MPI_STATUSES_IGNORE);

        // Copy into recvbuf
    }

    return ierr;
}


int MPIX_Request_free(MPIX_Request* request)
{
    if (request->local_L_n_msgs)
        free(request->local_L_requests);
    if (request->local_S_n_msgs)
        free(request->local_S_requests);
    if (request->local_R_n_msgs)
        free(request->local_R_requests);
    if (request->global_n_msgs)
        free(request->global_requests);

    // If Locality-Aware
    if (request->locality)
        destroy_locality_comm(request->locality);

    free(request);

    return 0;
}

