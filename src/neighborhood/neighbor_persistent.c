#include "neighbor.h"
#include "neighbor_persistent.h"

// Starting locality-aware requests
// 1. Start Local_L
// 2. Start and wait for local_S
// 3. Start global
int neighbor_start(MPIX_Request* request)
{
    if (request == NULL)
        return 0;

    int ierr = 0;
    int idx;

    char* send_buffer = NULL;
    int recv_size = 0;
    if (request->recv_size)
    {
        send_buffer = (char*)(request->sendbuf);
        recv_size = request->recv_size;
    }

    // Local L sends sendbuf
    if (request->local_L_n_msgs)
    {
        for (int i = 0; i < request->locality->local_L_comm->send_data->size_msgs; i++)
        {
            idx = request->locality->local_L_comm->send_data->indices[i];
            for (int j = 0; j < recv_size; j++)
                request->locality->local_L_comm->send_data->buffer[i*recv_size+j] = send_buffer[idx*recv_size+j];
        }
        ierr += MPI_Startall(request->local_L_n_msgs, request->local_L_requests);
    }



    // Local S sends sendbuf
    if (request->local_S_n_msgs)
    {
        for (int i = 0; i < request->locality->local_S_comm->send_data->size_msgs; i++)
        {
            idx = request->locality->local_S_comm->send_data->indices[i];

            for (int j = 0; j < recv_size; j++)
                request->locality->local_S_comm->send_data->buffer[i*recv_size+j] = send_buffer[idx*recv_size+j];
        }

        ierr += MPI_Startall(request->local_S_n_msgs, request->local_S_requests);
        ierr += MPI_Waitall(request->local_S_n_msgs, request->local_S_requests, MPI_STATUSES_IGNORE);


        // Copy into global->send_data->buffer
        for (int i = 0; i < request->locality->global_comm->send_data->size_msgs; i++)
        {
            idx = request->locality->global_comm->send_data->indices[i];
            for (int j = 0; j < recv_size; j++)
                request->locality->global_comm->send_data->buffer[i*recv_size+j] = request->locality->local_S_comm->recv_data->buffer[idx*recv_size+j];
        }
    }

    // Global sends buffer in locality, sendbuf in standard
    if (request->global_n_msgs)
        ierr += MPI_Startall(request->global_n_msgs, request->global_requests);

    return ierr;
}


// Wait for locality-aware requests
// 1. Wait for global
// 2. Start and wait for local_R
// 3. Wait for local_L
// TODO : Currently ignores the status!
int neighbor_wait(MPIX_Request* request, MPI_Status* status)
{
    if (request == NULL)
        return 0;

    int ierr = 0;
    int idx;

    char* recv_buffer = NULL;
    int recv_size = 0;
    if (request->recv_size)
    {
        recv_buffer = (char*)(request->recvbuf); 
        recv_size = request->recv_size;
    }


    // Global waits for recvs
    if (request->global_n_msgs)
    {
        ierr += MPI_Waitall(request->global_n_msgs, request->global_requests, MPI_STATUSES_IGNORE);

        if (request->local_R_n_msgs)
        {
            for (int i = 0; i < request->locality->local_R_comm->send_data->size_msgs; i++)
            {
                idx = request->locality->local_R_comm->send_data->indices[i];
                for (int j = 0; j < recv_size; j++)
                    request->locality->local_R_comm->send_data->buffer[i*recv_size+j] = request->locality->global_comm->recv_data->buffer[idx*recv_size+j];
            }
        }
    }

    // Wait for local_R recvs
    if (request->local_R_n_msgs)
    {
        ierr += MPI_Startall(request->local_R_n_msgs, request->local_R_requests);
        ierr += MPI_Waitall(request->local_R_n_msgs, request->local_R_requests, MPI_STATUSES_IGNORE);

        for (int i = 0; i < request->locality->local_R_comm->recv_data->size_msgs; i++)
        {
            idx = request->locality->local_R_comm->recv_data->indices[i];
            for (int j = 0; j < recv_size; j++)
                recv_buffer[idx*recv_size+j] = request->locality->local_R_comm->recv_data->buffer[i*recv_size+j];
        }
    }

    // Wait for local_L recvs
    if (request->local_L_n_msgs)
    {
        ierr += MPI_Waitall(request->local_L_n_msgs, request->local_L_requests, MPI_STATUSES_IGNORE);

        for (int i = 0; i < request->locality->local_L_comm->recv_data->size_msgs; i++)
        {
            idx = request->locality->local_L_comm->recv_data->indices[i];
            for (int j = 0; j < recv_size; j++)
                recv_buffer[idx*recv_size+j] = request->locality->local_L_comm->recv_data->buffer[i*recv_size+j];
        }
    }


    return ierr;
}


void init_neighbor_request(MPIX_Request** request_ptr)
{
    init_request(request_ptr);
    MPIX_Request* request = *request_ptr;

    request->start_function = (void*) neighbor_start;
    request->wait_function = (void*) neighbor_wait;
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
    int ierr = 0;
    int start, size;
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
        MPIX_Info* info,
        MPIX_Request** request_ptr)
{
    MPIX_Request* request;
    init_neighbor_request(&request);

    int ierr = 0;
    int tag = 349526;

    int indegree, outdegree, weighted;
    ierr += MPI_Dist_graph_neighbors_count(
            comm->neighbor_comm, 
            &indegree, 
            &outdegree, 
            &weighted);

    int* sources = (int*)malloc(indegree*sizeof(int));
    int* sourceweights = (int*)malloc(indegree*sizeof(int));
    int* destinations = (int*)malloc(outdegree*sizeof(int));
    int* destweights = (int*)malloc(outdegree*sizeof(int));

    ierr += MPI_Dist_graph_neighbors(
            comm->neighbor_comm, 
            indegree, 
            sources, 
            sourceweights,
            outdegree, 
            destinations, 
            destweights);

    request->global_n_msgs = indegree+outdegree;
    allocate_requests(request->global_n_msgs, &(request->global_requests));

    const char* send_buffer = (const char*)(sendbuffer);
    char* recv_buffer = (char*)(recvbuffer);
    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    for (int i = 0; i < indegree; i++)
    {
        ierr += MPI_Recv_init(&(recv_buffer[rdispls[i]*recv_size]), 
                recvcounts[i], 
                recvtype, 
                sources[i],
                tag,
                comm->neighbor_comm, 
                &(request->global_requests[i]));
    }

    for (int i = 0; i < outdegree; i++)
    {
        ierr += MPI_Send_init(&(send_buffer[sdispls[i]*send_size]),
                sendcounts[i],
                sendtype,
                destinations[i],
                tag,
                comm->neighbor_comm,
                &(request->global_requests[indegree+i]));
    }


    free(sources);
    free(sourceweights);
    free(destinations);
    free(destweights);

    *request_ptr = request;

    return ierr;
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
        MPIX_Info* info,
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

    int* sources = (int*)malloc(indegree*sizeof(int));
    int* sourceweights = (int*)malloc(indegree*sizeof(int));
    int* destinations = (int*)malloc(outdegree*sizeof(int));
    int* destweights = (int*)malloc(outdegree*sizeof(int));
    ierr += MPI_Dist_graph_neighbors(
            comm->neighbor_comm, 
            indegree, 
            sources, 
            sourceweights,
            outdegree, 
            destinations,
            destweights);

    MPIX_Request* request;
    init_neighbor_request(&request);

    request->global_n_msgs = indegree+outdegree;
    allocate_requests(request->global_n_msgs, &(request->global_requests));

    const char* send_buffer = (const char*)(sendbuffer);
    char* recv_buffer = (char*)(recvbuffer);

    for (int i = 0; i < outdegree; i++)
    {
        ierr += MPI_Send_init(&(send_buffer[sdispls[i]]),
                sendcounts[i],
                sendtypes[i],
                destinations[i],
                tag,
                comm->neighbor_comm,
                &(request->global_requests[indegree+i]));

    }
    for (int i = 0; i < indegree; i++)
    {
        ierr += MPI_Recv_init(&(recv_buffer[rdispls[i]]),
                recvcounts[i], 
                recvtypes[i], 
                sources[i],
                tag,
                comm->neighbor_comm, 
                &(request->global_requests[i]));

    }

    free(sources);
    free(sourceweights);
    free(destinations);
    free(destweights);

    *request_ptr = request;

    return ierr;
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
        MPIX_Info* info,
        MPIX_Request** request_ptr)
{
    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

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
    init_neighbor_request(&request);

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
    //init_communication(sendbuffer,
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


    return MPI_SUCCESS;
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
        MPIX_Info* info,
        MPIX_Request** request_ptr)
{
    int rank; 
    MPI_Comm_rank(comm->global_comm, &rank);

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

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

    int* global_sdispls = NULL;
    int* global_rdispls = NULL;

    int ctr;

    if (indegree)
    {
        sources = (int*)malloc(indegree*sizeof(int));
        sourceweights = (int*)malloc(indegree*sizeof(int));
        global_rdispls = (int*)malloc(indegree*sizeof(int));
        ctr = 0;
        for (int i = 0; i < indegree; i++)
        {
            global_rdispls[i] = ctr;
            ctr += recvcounts[i];
        }
    }

    if (outdegree)
    {
        destinations = (int*)malloc(outdegree*sizeof(int));
        destweights = (int*)malloc(outdegree*sizeof(int));
        global_sdispls = (int*)malloc(outdegree*sizeof(int));
        ctr = 0;
        for (int i = 0; i < outdegree; i++)
        {
            global_sdispls[i] = ctr;
            ctr += sendcounts[i];
        }

    }

    MPI_Dist_graph_neighbors(
            comm->neighbor_comm, 
            indegree, 
            sources, 
            sourceweights,
            outdegree, 
            destinations, 
            destweights);

    long send_size = 0;
    long recv_size = 0;
    for (int i = 0; i < indegree; i++)
        recv_size += recvcounts[i];
    for (int i = 0; i < outdegree; i++)
        send_size += sendcounts[i];

    long first_send;
    MPI_Exscan(&send_size, &first_send, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) first_send = 0;

    long* global_send_indices = NULL;
    long* global_recv_indices = NULL;

    if (recv_size)
        global_recv_indices = (long*)malloc(recv_size*sizeof(long));
    if (send_size)
        global_send_indices = (long*)malloc(send_size*sizeof(long));
    for (int i = 0; i < send_size; i++)
        global_send_indices[i] = first_send + i;

    MPIX_Neighbor_alltoallv(global_send_indices, sendcounts, global_sdispls, MPI_LONG, 
            global_recv_indices, recvcounts, global_rdispls, MPI_LONG, comm);

    int err = MPIX_Neighbor_locality_alltoallv_init(sendbuffer, sendcounts, sdispls, 
            global_send_indices, sendtype, recvbuffer, recvcounts, rdispls, 
            global_recv_indices, recvtype, comm, info, request_ptr);

    free(global_send_indices);
    free(global_recv_indices);

    free(global_sdispls);
    free(global_rdispls);

    free(sources);
    free(sourceweights);
    free(destinations);
    free(destweights);


    return MPI_SUCCESS;
}
