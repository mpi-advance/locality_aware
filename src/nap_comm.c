#include "nap_comm.h"

// MPI Dist Graph Create Adjacent Wrapper
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
    int* sourceweights = (int*)malloc(indegree*sizeof(int));
    for (i = 0; i < indegree; i++)
    {
        start = source_indptr[i];
        end = source_indptr[i+1];
        sourceweights[i] = (end - start);
    }
    int* destweights = (int*)malloc(outdegree*sizeof(int));
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

    return 0;
}


