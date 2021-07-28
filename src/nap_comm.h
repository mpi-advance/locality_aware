#ifndef NAPCOMM_HPP
#define NAPCOMM_HPP

#include <mpi.h>
#include <stdlib.h>

typedef struct _MPIX_Comm
{
    MPI_Comm comm;
    void* nap_comm;
} MPIX_Comm;

// Declarations of C++ methods
#ifdef __cplusplus
extern "C"
{
#endif
void MPIX_NAPinit(const int indegree, const int* sources, const int* source_indptr, const int* source_indices,
        const int outdegree, const int* destinations, const int* dest_indptr,
        const int* global_source_indices, const int* global_dest_indices,
        const MPI_Comm old_comm, void** nap_comm_ptr);

void MPIX_INAPsend(const void* buf, void* nap_comm,
        MPI_Datatype datatype, int tag,
        MPI_Comm comm);

void MPIX_INAPrecv(void* buf, void* nap_comm,
        MPI_Datatype datatype, int tag,
        MPI_Comm comm);

void MPIX_NAPwait(void* nap_comm);

void MPIX_NAPDestroy(void* nap_comm_ptr);


// MPI Dist Graph Create Adjacent Wrapper
int MPIX_Dist_graph_create_adjacent(
        MPI_Comm comm_old,
        int indegree,
        const int* sources,
        const int* source_indptr,
        const int* source_indices,
        const int* global_source_indices,
        int outdegree,
        const int* destinations,
        const int* dest_indptr,
        const int* global_dest_indices,
        MPI_Info info,
        int reorder,
        MPIX_Comm** comm_dist_graph);


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
        MPIX_Comm* comm);
     

int MPIX_Comm_free(MPIX_Comm* comm);

#ifdef __cplusplus
}
#endif

#endif
