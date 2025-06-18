#include "allreduce.h"
#include <string.h>
#include <math.h>

#ifdef GPU
#include "heterogeneous/gpu_allreduce.h"
#endif


int MPIX_Allreduce(const void* sendbuf,
        void* recvbuf, 
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm* comm)
{
#ifdef GPU
#ifdef GPU_AWARE
    return gpu_aware_allreduce_std(const void* sendbuf, void* recvbuf, count,
            datatype, op, comm);
#endif
#endif

    return MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm->global_comm);
}


// TODO : Assuming
int allreduce_lane(const void* sendbuf,
        void* recvbuf, 
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm* comm)
{
#ifdef GPU
#ifdef GPU_AWARE
    return gpu_aware_allreduce_lane(const void* sendbuf, void* recvbuf, count,
            datatype, op, comm);
#endif
#endif

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    int ppn, local_rank;
    MPI_Comm_size(comm->local_comm, &ppn);
    MPI_Comm_rank(comm->local_comm, &local_rank);

    // Reduce-Scatter On Node
    int* recvcounts = (int*)malloc(ppn*sizeof(int));
    int* displs = (int*)malloc((ppn+1)*sizeof(int));
    char* recv_buffer = (char*)recvbuf;

    int local_count = count / ppn;
    int first_local = local_rank * local_count;
    int extra = count % ppn;
    displs[0] = 0;
    for (int i = 0; i < ppn; i++)
    {
        recvcounts[i] = local_count;
        if (extra > i)
            recvcounts[i]++;
        displs[i+1] = displs[i] + recvcounts[i];
    }
    if (extra > local_rank)
    {
        local_count++;
        first_local += local_rank;
    }
    else
        first_local += extra;
    MPI_Reduce_scatter(sendbuf, &(recv_buffer[first_local]), recvcounts, datatype, op, comm->local_comm);


    // Allreduce between nodes
    MPI_Allreduce(MPI_IN_PLACE, &(recv_buffer[first_local]), local_count, datatype, op, comm->group_comm);

    // MPI_Gatherv On Node
    MPI_Allgatherv(MPI_IN_PLACE, local_count, datatype, recv_buffer, recvcounts, displs, datatype, 
            comm->local_comm);

    free(displs);
    free(recvcounts);

    return MPI_SUCCESS;
}

int allreduce_loc(const void* sendbuf,
        void* recvbuf, 
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPIX_Comm* comm)
{
#ifdef GPU
#ifdef GPU_AWARE
    return gpu_aware_allreduce_loc(const void* sendbuf, void* recvbuf, count,
            datatype, op, comm);
#endif
#endif

    if (comm->local_comm == MPI_COMM_NULL)
        MPIX_Comm_topo_init(comm);

    // Inter-Node allreduce (on group)
    MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm->group_comm);
    
    // Intra-Node allreduce
    MPI_Allreduce(MPI_IN_PLACE, recvbuf, count, datatype, op, comm->group_comm);

    return MPI_SUCCESS;
}

