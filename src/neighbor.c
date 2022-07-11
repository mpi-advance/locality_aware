#include "nap_comm.h"

// Leave MPI_Dist_graph_create_adjacent untouched... just creates a neighbor topolgoy

struct _MPIX_Request{
    int local_L_n_msgs;
    int local_S_n_msgs;
    int local_R_n_msgs;
    int global_n_msgs;

    MPI_Request* local_L_requests;
    MPI_Request* local_S_requests;
    MPI_Request* local_R_requests;
    MPI_Request* global_requests;
} MPIX_Request;


// Dist graph create initializes comm_dist_graph,
// and otherwise leaves method untouched
int MPIX_Dist_graph_create_adjacent(
        MPI_Comm comm_old,
        int indegree,
        const int sources[],
        const int sourceweights[],
        int outdegree,
        const int destinations[],
        const int destweights[],
        MPI_Info info,
        int reorder,
        MPIX_Comm** comm_dist_graph_ptr)
{
    MPIX_Comm* comm_dist_graph = (MPIX_Comm*)malloc(sizeof(MPIX_Comm));
    int err = MPI_Dist_graph_create_adjacent(comm_old, indegree, sources, sourceweights, 
            outdegree, destinations, destweights, info, reorder, &(comm_dist_graph->comm));
    *comm_dist_graph_ptr = comm_dist_graph;

    return err;
}

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
    int indegree, outdegree, weighted;
    MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted);

    int sources[indegree];
    int sourceweights[indegree];
    int destinations[outdegree];
    int destweights[outdegree];
    MPI_Dist_graph_neighbors(comm, indegree, sources, sourceweights,
            outdegree, destinations, destweights);

    MPIX_Request* request = (MPIX_Request*)malloc(sizeof(MPIX_Request));
    request->local_L_n_msgs = 0;
    request->local_S_n_msgs = 0;
    request->local_R_n_msgs = 0;
    request->global_n_msgs = indegree+outdegree;
    request->local_L_requests = NULL;
    request->local_S_requests = NULL;
    request->local_R_requests = NULL;
    request->global_requests = (MPI_Request*)malloc(sizeof(request->global_n_msgs));

    for (int i = 0; i < indegree; i++)
    {
        MPI_Recv_init(&(recvbuf[rdispls[i]]), &(recvcounts[i]), recvtype, 
                sources[i], tag, comm, &(request->global_requests[i]));
    }

    for (int i = 0; i < outdegree; i++)
    {
        MPI_Send_init(&(sendbuf[sdispls[i]]), &(sendcounts[i]), sendtype,
                destinations[i], tag, comm, &(request->global_requests[indegree+i]));
    }
}

// MPIX_Neighbor_alltoallv_init original arguments:
// - calls all MPI_Isend_init and MPI_Irecv_init

// MPIX_Neighbor_alltoallv_init with global indices:
// - initializes locality-aware neighbor alltoallv communication
// - calls all MPI_Isend_init and MPI_Irecv_init functions for
//      1. local_L communication
//      2. local_S communication
//      3. local_R communication
//      4. global_communication 
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
        MPIX_Request* request)
{
    int indegree, outdegree, weighted;
    MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted);

    int sources[indegree];
    int sourceweights[indegree];
    int destinations[outdegree];
    int destweights[outdegree];
    MPI_Dist_graph_neighbors(comm, indegree, sources, sourceweights,
            outdegree, destinations, destweights);

    // Initialize Locality-Aware Communication Strategy (3-Step)
    // E.G. Determine which processes talk to eachother at every step
    MPIX_NAPinit(
            outdegree, 
            destinations, 
            sdispls, 
            dest_indices,
            indegree, 
            sources, 
            rdispls,
            global_dest_indices, global_source_indices,
            comm_old, &(comm_dist_graph->nap_comm));

    // Initialize S-Sends
    n_recvs = nap_comm->local_S_comm->recv_data->num_msgs;
    recv_procs = nap_comm->local_S_comm->recv_data->procs;
    recv_starts = nap_comm->local_S_comm->recv_data->indptr;
    for (int i = 0; i < n_recvs; i++)
    {
        // TODO : buf, count, proc, local_S_tag, local_comm
        MPI_Recv_init(buf, 
                recv_starts[i+1] - recv_starts[i], 
                recvtype, 
                recv_procs[i], 
                nap_comm->local_S_comm->tag, 
                nap_comm->local_comm, 
                &(request->local_S_requests[i]));    
    }

    n_sends = nap_comm->local_S_comm->send_data->num_msgs;
    send_procs = nap_comm->local_S_comm->send_data->procs;
    send_starts = nap_comm->local_S_comm->send_data->indptr;
    send_indices = nap_comm->local_S_comm->send_data->indices;
    for (int i = 0; i < n_sends; i++)
    {
        MPI_Send_init(buf, 
                count, 
                sendtype, 
                proc, 
                local_S_tag, 
                local_comm, 
                &(request->local_S_requests[n_recv+i])); 
    }
    
    // Initialize S-Recvs


    for (int i = 0; i < indegree; i++)
    {
        MPI_Recv_init(&(recvbuf[rdispls[i]]), &(recvcounts[i]), recvtype, 
                sources[i], tag, comm, &(request->global_requests[i]));
    }

    for (int i = 0; i < outdegree; i++)
    {
        MPI_Send_init(&(sendbuf[sdispls[i]]), &(sendcounts[i]), sendtype,
                destinations[i], tag, comm, &(request->global_requests[indegree+i]));
    }



    nap_comm->global_comm
    // Initialize G-Sends

    // Initialize G-Recvs

    nap_comm->local_R_comm
    // Initialize R-sends

    // Initialize R-recvs

    nap_comm->local_L_comm
    // Initialize L-sends

    // Initialize L-recvs



    return 0;
}

int MPIX_Start(
        MPIX_Request* request)
{
}

int MPIX_Dist_graph_create_adjacent(
        MPI_Comm comm_old,
        int indegree,
        const int* sources,
        const int* source_indptr,
        const int* global_source_indices,
        int outdegree,
        const int* destinations,
        const int* dest_indptr,
        const int* dest_indices,
        const int* global_dest_indices,
        MPI_Info info,
        int reorder,
        MPIX_Comm** comm_dist_graph_ptr)
{
    MPIX_Comm* comm_dist_graph = (MPIX_Comm*)malloc(sizeof(MPIX_Comm));

    int i;
    int start, end;
    int* sourceweights = (int*)malloc((indegree+1)*sizeof(int));
    sourceweights[0] = 0;
    for (i = 0; i < indegree; i++)
    {
        start = source_indptr[i];
        end = source_indptr[i+1];
        sourceweights[i] = (end - start);
    }
    int* destweights = (int*)malloc((outdegree+1)*sizeof(int));
    destweights[0] = 0;
    for (i = 0; i < outdegree; i++)
    {
        start = dest_indptr[i];
        end = dest_indptr[i+1];
        destweights[i] = (end - start);
    }

    int err = MPI_Dist_graph_create_adjacent(comm_old, indegree, sources, sourceweights, 
            outdegree, destinations, destweights, info, reorder, &(comm_dist_graph->comm));
    
    MPIX_NAPinit(
            outdegree, destinations, dest_indptr, dest_indices,
            indegree, sources, source_indptr,
            global_dest_indices, global_source_indices,
            comm_old, &(comm_dist_graph->nap_comm));

    free(sourceweights);
    free(destweights);

    *comm_dist_graph_ptr = comm_dist_graph;

    return err;
}


// MPI Neighbor Alltoallv Wrapper
// TODO -- assumes one single element is sent for each index in 
//          dist_graph_create_adjacent
int MPIX_Neighbor_alltoallv(
        const void* sendbuf,
        const int* sendcounts,
        const int* send_indptr,
        MPI_Datatype sendtype,
        void* recvbuf, 
        const int* recvcounts,
        const int* recv_indptr,
        MPI_Datatype recvtype,
        MPIX_Comm* comm)
{
    // Initialize all recvs
    MPIX_INAPrecv(recvbuf, comm->nap_comm, recvtype, 14528, comm->comm);

    // Initialize all sends
    MPIX_INAPsend(sendbuf, comm->nap_comm, sendtype, 14528, comm->comm);

    // Wait for all sends and recvs to complete
    MPIX_NAPwait(comm->nap_comm);

    return 0;
}

int MPIX_Comm_free(MPIX_Comm* comm_dist_graph)
{
    MPI_Comm_free(&(comm_dist_graph->comm));
    MPIX_NAPDestroy(comm_dist_graph->nap_comm);
    free(comm_dist_graph);

    return 0;
}


