#include "locality_aware.h" // #TODO -- figure out why this is needed
#include "persistent/MPIL_Request.h"
#include "communicator/MPIL_Comm.h"
#include "neighborhood/neighborhood_init.h"
#include "neighborhood/MPIL_Topo.h"

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

    // Initialize Locality-Aware Communication Strategy (3-Step)
    // E.G. Determine which processes talk to each other at every step
    // TODO : instead of mpi_comm, use comm
    //        - will need to create local_comm in dist_graph_create_adjacent...
    init_locality(topo->outdegree,
                  topo->destinations,
                  sdispls,
                  sendcounts,
                  topo->indegree,
                  topo->sources,
                  rdispls,
                  recvcounts,
                  global_sindices,
                  global_rindices,
                  sendtype,
                  recvtype,
                  comm,  // communicator used in dist_graph_create_adjacent
                  request);

    request->sendbuf = sendbuffer;
    request->recvbuf = recvbuffer;
    MPI_Type_size(recvtype, &(request->recv_size));

    // Local L Communication
    // init_communication(sendbuffer,
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

    *request_ptr = request;

    return MPI_SUCCESS;
}
