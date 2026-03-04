#include <stdlib.h>

#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"
#include "neighborhood/MPIL_Topo.h"
#include "neighborhood/neighborhood_init.h"
#include "persistent/MPIL_Request.h"

// Locality-Aware Extension to Persistent Neighbor Alltoallv
// Needs global indices for each send and receive
int neighbor_alltoallv_init_locality_ext(const void* sendbuffer,
                                         const int sendcounts[],
                                         const int sdispls[],
                                         const long global_sindices[],
                                         MPI_Datatype sendtype,
                                         void* recvbuffer,
                                         const int recvcounts[],
                                         const int rdispls[],
                                         const long global_rindices[],
                                         MPI_Datatype recvtype,
                                         MPIL_Topo* topo,
                                         MPIL_Comm* comm,
                                         MPIL_Info* info,
                                         MPIL_Request** request_ptr)
{
    if (comm->local_comm == MPI_COMM_NULL)
    {
        MPIL_Comm_topo_init(comm);
    }

    MPIL_Request* request;
    init_neighbor_request(&request);
    init_neighbor_request(&(request->local_L_request));
    init_neighbor_request(&(request->local_S_request));
    init_neighbor_request(&(request->local_R_request));

    int indegree  = 0;
    int outdegree = 0;

    int* sources       = NULL;
    int* source_counts = NULL;
    int* source_displs = NULL;
    int* destinations  = NULL;
    int* dest_counts   = NULL;
    int* dest_displs   = NULL;

    if (topo->indegree)
    {
        sources       = (int*)malloc(topo->indegree * sizeof(int));
        source_counts = (int*)malloc(topo->indegree * sizeof(int));
        source_displs = (int*)malloc(topo->indegree * sizeof(int));
    }
    if (topo->outdegree)
    {
        destinations = (int*)malloc(topo->outdegree * sizeof(int));
        dest_counts  = (int*)malloc(topo->outdegree * sizeof(int));
        dest_displs  = (int*)malloc(topo->outdegree * sizeof(int));
    }

    for (int i = 0; i < topo->indegree; i++)
    {
        if (recvcounts[i])
        {
            sources[indegree]       = topo->sources[i];
            source_counts[indegree] = recvcounts[i];
            source_displs[indegree] = rdispls[i];
            indegree++;
        }
    }

    for (int i = 0; i < topo->outdegree; i++)
    {
        if (sendcounts[i])
        {
            destinations[outdegree] = topo->destinations[i];
            dest_counts[outdegree]  = sendcounts[i];
            dest_displs[outdegree]  = sdispls[i];
            outdegree++;
        }
    }

    LocalityComm* locality;
    init_locality_comm(&locality, comm, sendtype, recvtype);
    

    // Initialize Locality-Aware Communication Strategy (3-Step)
    // E.G. Determine which processes talk to each other at every step
    // TODO : instead of mpi_comm, use comm
    //        - will need to create local_comm in dist_graph_create_adjacent...
    init_locality(outdegree,
                  destinations,
                  dest_displs,
                  dest_counts,
                  indegree,
                  sources,
                  source_displs,
                  source_counts,
                  global_sindices,
                  global_rindices,
                  sendtype,
                  recvtype,
                  comm,  // communicator used in dist_graph_create_adjacent
                  request,
                  locality);

//    request->sendbuf = sendbuffer;
//    request->recvbuf = recvbuffer;
    int send_size, recv_size;
    MPI_Type_size(sendtype, &(send_size));
    MPI_Type_size(recvtype, &(recv_size));

    // Initialize packing buffers for Local_L
    init_packing_buffers(request->local_L_request,
                            locality->local_L_comm->send_data->size_msgs,
                            locality->local_L_comm->send_data->indices,
                            send_size,
                            sendbuffer,
                            locality->local_L_comm->recv_data->size_msgs,
                            locality->local_L_comm->recv_data->indices,
                            recv_size,
                            recvbuffer);

    // Initialize packing buffers for Local_S
    init_packing_buffers(request->local_S_request,
                            locality->local_S_comm->send_data->size_msgs,
                            locality->local_S_comm->send_data->indices,
                            send_size,
                            sendbuffer,
                            locality->local_S_comm->recv_data->size_msgs,
                            NULL,
                            send_size,
                            NULL);

    // Initialize packing buffers for global
    init_packing_buffers(request,
                            locality->global_comm->send_data->size_msgs,
                            locality->global_comm->send_data->indices,
                            send_size,
                            request->local_S_request->tmp_recvbuf,
                            locality->global_comm->recv_data->size_msgs,
                            NULL,
                            recv_size,
                            NULL);

    // Initialize packing buffers for Local_R
    init_packing_buffers(request->local_R_request,
                            locality->local_R_comm->send_data->size_msgs,
                            locality->local_R_comm->send_data->indices,
                            recv_size,
                            request->tmp_recvbuf,
                            locality->local_R_comm->recv_data->size_msgs,
                            locality->local_R_comm->recv_data->indices,
                            recv_size,
                            recvbuffer);





    MPIL_Info* mpil_info;
    MPIL_Info_init(&mpil_info);
    int tag;

    // Local L Communication
    // init_communication(sendbuffer,
    MPIL_Topo* topo_step;
    MPIL_Topo_init(locality->local_L_comm->send_data->num_msgs, 
                    locality->local_L_comm->send_data->procs,
                    MPI_UNWEIGHTED,
                    locality->local_L_comm->recv_data->num_msgs,
                    locality->local_L_comm->recv_data->procs,
                    MPI_UNWEIGHTED,
                    mpil_info,
                    &topo_step);
    MPIL_Comm_tag(comm, &tag);
    neighbor_alltoallv_init_standard_helper(
                        request->local_L_request->tmp_sendbuf,
                        locality->local_L_comm->send_data->counts,
                        locality->local_L_comm->send_data->indptr,
                        sendtype,
                        request->local_L_request->tmp_recvbuf,
                        locality->local_L_comm->recv_data->counts,
                        locality->local_L_comm->recv_data->indptr,
                        recvtype,
                        topo_step,
                        comm->local_comm,
                        mpil_info,
                        tag,
                        request->local_L_request);
    MPIL_Topo_free(&topo_step);
                        

    // Local S Communication
    MPIL_Topo_init(locality->local_S_comm->send_data->num_msgs, 
                    locality->local_S_comm->send_data->procs,
                    MPI_UNWEIGHTED,
                    locality->local_S_comm->recv_data->num_msgs,
                    locality->local_S_comm->recv_data->procs,
                    MPI_UNWEIGHTED,
                    mpil_info,
                    &topo_step);
    MPIL_Comm_tag(comm, &tag);
    neighbor_alltoallv_init_standard_helper(
                        request->local_S_request->tmp_sendbuf,
                        locality->local_S_comm->send_data->counts,
                        locality->local_S_comm->send_data->indptr,
                        sendtype,
                        request->local_S_request->tmp_recvbuf,
                        locality->local_S_comm->recv_data->counts,
                        locality->local_S_comm->recv_data->indptr,
                        recvtype,
                        topo_step,
                        comm->local_comm,
                        mpil_info,
                        tag,
                        request->local_S_request);
    MPIL_Topo_free(&topo_step);
    

    // Global Communication
    MPIL_Topo_init(locality->global_comm->send_data->num_msgs, 
                    locality->global_comm->send_data->procs,
                    MPI_UNWEIGHTED,
                    locality->global_comm->recv_data->num_msgs,
                    locality->global_comm->recv_data->procs,
                    MPI_UNWEIGHTED,
                    mpil_info,
                    &topo_step);
    MPIL_Comm_tag(comm, &tag);
    neighbor_alltoallv_init_standard_helper(
                        request->tmp_sendbuf,
                        locality->global_comm->send_data->counts,
                        locality->global_comm->send_data->indptr,
                        sendtype,
                        request->tmp_recvbuf,
                        locality->global_comm->recv_data->counts,
                        locality->global_comm->recv_data->indptr,
                        recvtype,
                        topo_step,
                        comm->global_comm,
                        mpil_info,
                        tag,
                        request);
    MPIL_Topo_free(&topo_step);


    // Local R Communication
    MPIL_Topo_init(locality->local_R_comm->send_data->num_msgs, 
                    locality->local_R_comm->send_data->procs,
                    MPI_UNWEIGHTED,
                    locality->local_R_comm->recv_data->num_msgs,
                    locality->local_R_comm->recv_data->procs,
                    MPI_UNWEIGHTED,
                    mpil_info,
                    &topo_step);
    MPIL_Comm_tag(comm, &tag);
    neighbor_alltoallv_init_standard_helper(
                        request->local_R_request->tmp_sendbuf,
                        locality->local_R_comm->send_data->counts,
                        locality->local_R_comm->send_data->indptr,
                        sendtype,
                        request->local_R_request->tmp_recvbuf,
                        locality->local_R_comm->recv_data->counts,
                        locality->local_R_comm->recv_data->indptr,
                        recvtype,
                        topo_step,
                        comm->local_comm,
                        mpil_info,
                        tag,
                        request->local_R_request);

    destroy_locality_comm(locality);
    MPIL_Info_free(&mpil_info);
    MPIL_Topo_free(&topo_step);
    

    *request_ptr = request;

    if (topo->indegree)
    {
        free(sources);
        free(source_counts);
        free(source_displs);
    }
    if (topo->outdegree)
    {
        free(destinations);
        free(dest_counts);
        free(dest_displs);
    }

    return MPI_SUCCESS;
}
