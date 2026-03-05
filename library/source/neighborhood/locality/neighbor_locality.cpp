#include "neighborhood/neighbor_locality.h"

#include <algorithm>

#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"
#include "persistent/MPIL_Request.h"
#include "neighborhood/MPIL_Topo.h"
#include "neighborhood/neighborhood_init.h"
#include "neighborhood/alltoall_crs.h"

void map_indices(CommData* idx_data, std::map<long, int>& global_map);
void map_indices(CommData* idx_data, const CommData* map_data);

/******************************************
 ****
 **** Main Methods
 ****
 ******************************************/
// Declarations of C++ methods
#ifdef __cplusplus
extern "C" {
#endif
// Initialize NAPComm* structure, to be used for any number of
// instances of communication
void init_locality(const int n_sends,
                   const int* send_procs,
                   const int* send_indptr,
                   const int* sendcounts,
                   const void* sendbuffer,
                   const int n_recvs,
                   const int* recv_procs,
                   const int* recv_indptr,
                   const int* recvcounts,
                   void* recvbuffer,
                   const long* global_send_indices,
                   const long* global_recv_indices,
                   const MPI_Datatype sendtype,
                   const MPI_Datatype recvtype,
                   MPIL_Comm* mpil_comm,
                   MPIL_Request* request)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(mpil_comm->global_comm, &rank);
    MPI_Comm_size(mpil_comm->global_comm, &num_procs);

    CommData* local_L_send_data = (CommData*)calloc(1, sizeof(CommData));
    CommData* local_L_recv_data = (CommData*)calloc(1, sizeof(CommData));
    CommData* local_S_send_data = (CommData*)calloc(1, sizeof(CommData));
    CommData* local_S_recv_data = (CommData*)calloc(1, sizeof(CommData));
    CommData* local_R_send_data = (CommData*)calloc(1, sizeof(CommData));
    CommData* local_R_recv_data = (CommData*)calloc(1, sizeof(CommData));
    CommData* global_send_data = (CommData*)calloc(1, sizeof(CommData));
    CommData* global_recv_data = (CommData*)calloc(1, sizeof(CommData));

    int send_size, recv_size;
    MPI_Type_size(sendtype, &(send_size));
    MPI_Type_size(recvtype, &(recv_size));

    // Find global send nodes
    std::vector<int> send_nodes;
    std::vector<int> send_node_to_local;
    map_procs_to_nodes(n_sends,
                       send_procs,
                       sendcounts,
                       send_nodes,
                       send_node_to_local,
                       true, 
                       mpil_comm);

    // Form initial send local comm
    std::vector<int> recv_idx_nodes;
    form_local_comm(n_sends,
                    send_procs,
                    send_indptr,
                    sendcounts,
                    global_send_indices,
                    send_node_to_local,
                    local_S_send_data,
                    local_S_recv_data,
                    local_L_send_data,
                    recv_idx_nodes,
                    mpil_comm);

    // Form global send data
    form_global_comm(local_S_recv_data,
                     global_send_data,
                     recv_idx_nodes,
                     mpil_comm);

    // Find global recv nodes
    std::vector<int> recv_nodes;
    std::vector<int> recv_node_to_local;
    map_procs_to_nodes(n_recvs,
                       recv_procs,
                       recvcounts,
                       recv_nodes,
                       recv_node_to_local,
                       false,
                       mpil_comm);

    // Form final recv local comm
    std::vector<int> send_idx_nodes;
    form_local_comm(n_recvs,
                    recv_procs,
                    recv_indptr,
                    recvcounts,
                    global_recv_indices,
                    recv_node_to_local,
                    local_R_recv_data,
                    local_R_send_data,
                    local_L_recv_data,
                    send_idx_nodes,
                    mpil_comm);

    // Form global recv data
    form_global_comm(local_R_send_data,
                     global_recv_data,
                     send_idx_nodes,
                     mpil_comm);

    // Update procs for global_comm send and recvs
    update_global_comm(global_send_data,
                        global_recv_data,
                        mpil_comm);

    // Update send and receive indices
    std::map<long, int> send_global_to_local;
    std::map<long, int> recv_global_to_local;
    int ctr = 0;
    int start, end;
    for (int i = 0; i < n_sends; i++)
    {
        start = send_indptr[i];
        end   = start + sendcounts[i];
        for (int j = start; j < end; j++)
        {
            send_global_to_local[global_send_indices[ctr++]] = j;
        }
    }

    ctr = 0;
    for (int i = 0; i < n_recvs; i++)
    {
        start = recv_indptr[i];
        end   = start + recvcounts[i];
        for (int j = start; j < end; j++)
        {
            recv_global_to_local[global_recv_indices[ctr++]] = j;
        }
    }

    remove_duplicates(local_S_send_data);
    remove_duplicates(local_S_recv_data);
    remove_duplicates(local_R_send_data);
    remove_duplicates(local_R_recv_data);
    remove_duplicates(global_send_data);
    remove_duplicates(global_recv_data);

    // Map global indices to usable indices
    map_indices(global_send_data, local_S_recv_data);
    map_indices(local_R_send_data, global_recv_data);
    map_indices(local_S_send_data, send_global_to_local);
    map_indices(local_L_send_data, send_global_to_local);
    map_indices(local_R_recv_data, recv_global_to_local);
    map_indices(local_L_recv_data, recv_global_to_local);

    // Don't need local_S or global recv indices (just contiguous)
    if (local_S_recv_data->indices)
    {
        free(local_S_recv_data->indices);
        local_S_recv_data->indices = NULL;
    }
    if (global_recv_data->indices)
    {
        free(global_recv_data->indices);
        global_recv_data->indices = NULL;
    }


    // Initialize packing buffers for Local_L
    init_packing_buffers(request->local_L_request,
                            local_L_send_data->size_msgs,
                            local_L_send_data->indices,
                            send_size,
                            sendbuffer,
                            local_L_recv_data->size_msgs,
                            local_L_recv_data->indices,
                            recv_size,
                            recvbuffer);

    // Initialize packing buffers for Local_S
    init_packing_buffers(request->local_S_request,
                            local_S_send_data->size_msgs,
                            local_S_send_data->indices,
                            send_size,
                            sendbuffer,
                            local_S_recv_data->size_msgs,
                            NULL,
                            send_size,
                            NULL);

    // Initialize packing buffers for global
    init_packing_buffers(request,
                            global_send_data->size_msgs,
                            global_send_data->indices,
                            send_size,
                            request->local_S_request->tmp_recvbuf,
                            global_recv_data->size_msgs,
                            NULL,
                            recv_size,
                            NULL);

    // Initialize packing buffers for Local_R
    init_packing_buffers(request->local_R_request,
                            local_R_send_data->size_msgs,
                            local_R_send_data->indices,
                            recv_size,
                            request->tmp_recvbuf,
                            local_R_recv_data->size_msgs,
                            local_R_recv_data->indices,
                            recv_size,
                            recvbuffer);





    MPIL_Info* mpil_info;
    MPIL_Info_init(&mpil_info);
    int tag;

    // Local L Communication
    MPIL_Topo* topo_step;
    MPIL_Topo_init(local_L_recv_data->num_msgs,
                    local_L_recv_data->procs,
                    MPI_UNWEIGHTED,
                    local_L_send_data->num_msgs, 
                    local_L_send_data->procs,
                    MPI_UNWEIGHTED,
                    mpil_info,
                    &topo_step);
    MPIL_Comm_tag(mpil_comm, &tag);
    neighbor_alltoallv_init_standard_helper(
                        request->local_L_request->tmp_sendbuf,
                        local_L_send_data->counts,
                        local_L_send_data->indptr,
                        sendtype,
                        request->local_L_request->tmp_recvbuf,
                        local_L_recv_data->counts,
                        local_L_recv_data->indptr,
                        recvtype,
                        topo_step,
                        mpil_comm->local_comm,
                        mpil_info,
                        tag,
                        request->local_L_request);
    MPIL_Topo_free(&topo_step);
                        

    // Local S Communication
    MPIL_Topo_init(local_S_recv_data->num_msgs,
                    local_S_recv_data->procs,
                    MPI_UNWEIGHTED,
                    local_S_send_data->num_msgs, 
                    local_S_send_data->procs,
                    MPI_UNWEIGHTED,
                    mpil_info,
                    &topo_step);
    MPIL_Comm_tag(mpil_comm, &tag);
    neighbor_alltoallv_init_standard_helper(
                        request->local_S_request->tmp_sendbuf,
                        local_S_send_data->counts,
                        local_S_send_data->indptr,
                        sendtype,
                        request->local_S_request->tmp_recvbuf,
                        local_S_recv_data->counts,
                        local_S_recv_data->indptr,
                        recvtype,
                        topo_step,
                        mpil_comm->local_comm,
                        mpil_info,
                        tag,
                        request->local_S_request);
    MPIL_Topo_free(&topo_step);
    

    // Global Communication
    MPIL_Topo_init(global_recv_data->num_msgs,
                    global_recv_data->procs,
                    MPI_UNWEIGHTED,
                    global_send_data->num_msgs, 
                    global_send_data->procs,
                    MPI_UNWEIGHTED,
                    mpil_info,
                    &topo_step);
    MPIL_Comm_tag(mpil_comm, &tag);
    neighbor_alltoallv_init_standard_helper(
                        request->tmp_sendbuf,
                        global_send_data->counts,
                        global_send_data->indptr,
                        sendtype,
                        request->tmp_recvbuf,
                        global_recv_data->counts,
                        global_recv_data->indptr,
                        recvtype,
                        topo_step,
                        mpil_comm->global_comm,
                        mpil_info,
                        tag,
                        request);
    MPIL_Topo_free(&topo_step);


    // Local R Communication
    MPIL_Topo_init(local_R_recv_data->num_msgs,
                    local_R_recv_data->procs,
                    MPI_UNWEIGHTED,
                    local_R_send_data->num_msgs, 
                    local_R_send_data->procs,
                    MPI_UNWEIGHTED,
                    mpil_info,
                    &topo_step);
    MPIL_Comm_tag(mpil_comm, &tag);
    neighbor_alltoallv_init_standard_helper(
                        request->local_R_request->tmp_sendbuf,
                        local_R_send_data->counts,
                        local_R_send_data->indptr,
                        sendtype,
                        request->local_R_request->tmp_recvbuf,
                        local_R_recv_data->counts,
                        local_R_recv_data->indptr,
                        recvtype,
                        topo_step,
                        mpil_comm->local_comm,
                        mpil_info,
                        tag,
                        request->local_R_request);

    destroy_comm_data(local_L_send_data);
    destroy_comm_data(local_L_recv_data);
    destroy_comm_data(local_S_send_data);
    destroy_comm_data(local_S_recv_data);
    destroy_comm_data(local_R_send_data);
    destroy_comm_data(local_R_recv_data);
    destroy_comm_data(global_send_data);
    destroy_comm_data(global_recv_data);

    MPIL_Info_free(&mpil_info);
    MPIL_Topo_free(&topo_step);

}
#ifdef __cplusplus
}
#endif

/******************************************
 ****
 **** Helper Methods
 ****
 ******************************************/
// Map original communication processes to nodes on which they lie
// And assign local processes to each node
void map_procs_to_nodes(const int orig_num_msgs,
                        const int* orig_procs,
                        const int* orig_counts,
                        std::vector<int>& msg_nodes,
                        std::vector<int>& msg_node_to_local,
                        bool incr,
                        MPIL_Comm* mpil_comm)
{
    int local_num_procs;
    MPI_Comm_size(mpil_comm->local_comm, &local_num_procs);

    int proc, size, node;
    int local_proc;
    int inc;
    std::vector<int> node_sizes;

    int num_nodes = mpil_comm->num_nodes;
    int rank_node = mpil_comm->rank_node;

    // Map local msg_procs to local msg_nodes
    node_sizes.resize(num_nodes, 0);
    for (int i = 0; i < orig_num_msgs; i++)
    {
        proc = orig_procs[i];
        size = orig_counts[i];
        node = get_node(mpil_comm, proc);
        node_sizes[node] += size;
    }

    // Gather all send nodes and sizes among ranks local to node
    MPI_Allreduce(MPI_IN_PLACE,
                  node_sizes.data(),
                  num_nodes,
                  MPI_INT,
                  MPI_SUM,
                  mpil_comm->local_comm);
    for (int i = 0; i < num_nodes; i++)
    {
        if (node_sizes[i] && i != rank_node)
        {
            msg_nodes.push_back(i);
        }
    }
    std::sort(msg_nodes.begin(), msg_nodes.end(), [&](const int i, const int j) {
        return node_sizes[i] > node_sizes[j];
    });

    // Map send_nodes to local ranks
    msg_node_to_local.resize(num_nodes, -1);
    if (incr)
    {
        local_proc = 0;
        inc        = 1;
    }
    else
    {
        local_proc = local_num_procs - 1;
        inc        = -1;
    }
    for (size_t i = 0; i < msg_nodes.size(); i++)
    {
        node                    = msg_nodes[i];
        msg_node_to_local[node] = local_proc;

        if (local_proc == local_num_procs - 1 && inc == 1)
        {
            inc = -1;
        }
        else if (local_proc == 0 && inc == -1)
        {
            inc = 1;
        }
        else
        {
            local_proc += inc;
        }
    }
}

// Form step of local communication (either initial local_S communicator
// or final local_L communicator) along with the corresponding portion
// of the fully local (local_L) communicator.
void form_local_comm(const int orig_num_sends,
                     const int* orig_send_procs,
                     const int* orig_send_ptr,
                     const int* orig_sendcounts,
                     const long* orig_send_indices,
                     const std::vector<int>& nodes_to_local,
                     CommData* send_data,
                     CommData* recv_data,
                     CommData* local_data,
                     std::vector<int>& recv_idx_nodes,
                     MPIL_Comm* mpil_comm)
{
    // MPI_Information
    int local_rank, local_num_procs;
    MPI_Comm_rank(mpil_comm->local_comm, &local_rank);
    MPI_Comm_size(mpil_comm->local_comm, &local_num_procs);

    // Declare variables
    int global_proc, local_proc;
    int size, ctr;
    int start, end, node;
    int idx, proc_idx;
    int global_idx;

    std::vector<int> send_sizes;
    std::vector<int> orig_to_node;
    std::vector<int> local_idx;

    // Initialize variables
    orig_to_node.resize(orig_num_sends);
    local_idx.resize(local_num_procs);
    send_sizes.resize(local_num_procs, 0);

    // Allocate sizes
    init_num_msgs(send_data, local_num_procs);
    init_num_msgs(local_data, local_num_procs);

    // Form local_S_comm
    send_data->num_msgs  = 0;
    local_data->num_msgs = 0;
    for (int i = 0; i < orig_num_sends; i++)
    {
        global_proc = orig_send_procs[i];
        size        = orig_sendcounts[i];
        node        = get_node(mpil_comm, global_proc);
        if (mpil_comm->rank_node != node)
        {
            local_proc = nodes_to_local[node];
            if (send_sizes[local_proc] == 0)
            {
                local_idx[local_proc]                   = send_data->num_msgs;
                send_data->procs[send_data->num_msgs++] = local_proc;
            }
            orig_to_node[i] = node;
            send_sizes[local_proc] += size;
        }
        else
        {
            orig_to_node[i] = -1;
            local_data->procs[local_data->num_msgs] =
                get_local_proc(mpil_comm, global_proc);
            local_data->counts[local_data->num_msgs] = size;
            local_data->size_msgs += size;
            local_data->num_msgs++;
            local_data->indptr[local_data->num_msgs] = local_data->size_msgs;
        }
    }
    init_size_msgs(local_data, local_data->size_msgs);

    for (int i = 0; i < send_data->num_msgs; i++)
    {
        local_proc               = send_data->procs[i];
        send_data->counts[i]     = send_sizes[local_proc];
        send_data->indptr[i + 1] = send_data->indptr[i] + send_sizes[local_proc];
        send_sizes[local_proc]   = 0;
    }
    send_data->size_msgs = send_data->indptr[send_data->num_msgs];

    // Allocate send_indices and fill vector
    init_size_msgs(send_data, send_data->size_msgs);

    std::vector<int> send_idx_node(send_data->size_msgs);
    local_data->size_msgs = 0;
    ctr                   = 0;
    for (int i = 0; i < orig_num_sends; i++)
    {
        node  = orig_to_node[i];
        start = orig_send_ptr[i];
        end   = orig_send_ptr[i] + orig_sendcounts[i];
        if (node == -1)
        {
            for (int j = start; j < end; j++)
            {
                global_idx                                   = orig_send_indices[ctr++];
                local_data->indices[local_data->size_msgs++] = global_idx;
            }
        }
        else
        {
            local_proc = nodes_to_local[node];
            proc_idx   = local_idx[local_proc];
            for (int j = start; j < end; j++)
            {
                global_idx = orig_send_indices[ctr++];
                idx        = send_data->indptr[proc_idx] + send_sizes[local_proc]++;
                send_data->indices[idx] = global_idx;
                send_idx_node[idx]      = node;
            }
        }
    }

    // Send 'local_S_comm send' info (to form local_S recv)
    std::vector<int> send_buf(2 * send_data->size_msgs);
    for (int i = 0; i < send_data->size_msgs; i++)
    {
        send_buf[2 * i]     = send_data->indices[i];
        send_buf[2 * i + 1] = send_idx_node[i];
    }

    // Dynamic comm uses mpil_comm->global_comm
    // So create new one with global_comm set to current local_comm
    MPIL_Comm* local_mpil_comm;
    MPIL_Comm_init(&local_mpil_comm, mpil_comm->local_comm);

    // Reseting local comm's tag to next available mpil_comm tag
    // So that calling this method multiple times doesn't result
    // in multiple dynamic comms on same tag
    get_tag(mpil_comm, &local_mpil_comm->tag);

    MPIL_Info* local_info;
    MPIL_Info_init(&local_info);

    int n_recvs, s_recvs;
    int *src, *recvcounts, *rdispls, *recv_buf;
    alltoallv_crs_personalized(send_data->num_msgs,
                       send_data->size_msgs,
                       send_data->procs,
                       send_data->counts,
                       send_data->indptr,
                       MPI_2INT,
                       send_buf.data(),
                       &n_recvs,
                       &s_recvs,
                       &src,
                       &recvcounts,
                       &rdispls,
                       MPI_2INT,
                       (void**)(&recv_buf),
                       local_info,
                       local_mpil_comm);

    init_num_msgs(recv_data, n_recvs);
    init_size_msgs(recv_data, s_recvs);
    memcpy(recv_data->procs,  src,    n_recvs * sizeof(int));
    memcpy(recv_data->counts, recvcounts, n_recvs * sizeof(int));
    memcpy(recv_data->indptr, rdispls, (n_recvs + 1) * sizeof(int));

    recv_idx_nodes.resize(s_recvs);
    int* recv_buf_int = (int*)recv_buf;
    for (int i = 0; i < s_recvs; i++)
    {
        recv_data->indices[i] = recv_buf_int[2 * i];
        recv_idx_nodes[i]     = recv_buf_int[2 * i + 1];
    }

    MPIL_Info_free(&local_info);
    MPIL_Comm_free(&local_mpil_comm);

    MPIL_Free(src);
    MPIL_Free(recvcounts);
    MPIL_Free(rdispls);
    MPIL_Free(recv_buf);
    

}

// Form portion of inter-node communication (data corresponding to
// either global send or global recv), with node id currently in
// place of process with which to communicate
void form_global_comm(CommData* local_data,
                      CommData* global_data,
                      std::vector<int>& local_data_nodes,
                      MPIL_Comm* mpil_comm)
{
    std::vector<int> node_sizes;
    std::vector<int> node_ctr;

    // Get MPI Information
    int num_nodes = mpil_comm->num_nodes;

    int node_idx, node;
    int start, end, idx;

    node_sizes.resize(num_nodes, 0);

    for (int i = 0; i < local_data->size_msgs; i++)
    {
        node = local_data_nodes[i];
        if (node_sizes[node] == 0)
        {
            global_data->num_msgs++;
        }
        node_sizes[node]++;
    }
    init_num_msgs(global_data, global_data->num_msgs);

    node_ctr.resize(global_data->num_msgs, 0);
    global_data->num_msgs  = 0;
    global_data->indptr[0] = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        if (node_sizes[i])
        {
            global_data->procs[global_data->num_msgs] = i;
            global_data->counts[global_data->num_msgs] = node_sizes[i];
            global_data->size_msgs += node_sizes[i];
            node_sizes[i] = global_data->num_msgs;
            global_data->num_msgs++;
            global_data->indptr[global_data->num_msgs] = global_data->size_msgs;
        }
    }

    init_size_msgs(global_data, global_data->size_msgs);
    for (int i = 0; i < local_data->num_msgs; i++)
    {
        start = local_data->indptr[i];
        end   = local_data->indptr[i + 1];
        for (int j = start; j < end; j++)
        {
            node     = local_data_nodes[j];
            node_idx = node_sizes[node];
            idx      = global_data->indptr[node_idx] + node_ctr[node_idx]++;
            global_data->indices[idx] = local_data->indices[j];
        }
    }
}

// Replace send and receive processes with the node id's currently in their place
void update_global_comm(CommData* global_send_data,
                        CommData* global_recv_data,
                        MPIL_Comm* mpil_comm)
{
    int local_rank;
    MPI_Comm_rank(mpil_comm->local_comm, &local_rank);
    int num_nodes = mpil_comm->num_nodes;

    std::vector<int> nodes(2*num_nodes, 0);

    MPIL_Info* mpil_info;
    MPIL_Info_init(&mpil_info);

    // Initialize send side for dynamic communication
    std::vector<int> dest(global_send_data->num_msgs);
    std::vector<int> vals(global_send_data->num_msgs, mpil_comm->rank_node);
    for (int i = 0; i < global_send_data->num_msgs; i++)
        dest[i] = get_global_proc(mpil_comm, global_send_data->procs[i], local_rank);

    int n_recvs;
    int *src, *recvbuf;
    MPIL_Alltoall_crs(global_send_data->num_msgs,
                    dest.data(), 
                    1,
                    MPI_INT,
                    vals.data(),
                    &n_recvs,
                    &src,
                    1,
                    MPI_INT,
                    (void**) &recvbuf,
                    mpil_info,
                    mpil_comm);
    for (int i = 0; i < n_recvs; i++)
        nodes[recvbuf[i]] = src[i];

    MPIL_Free(src);
    MPIL_Free(recvbuf);

    dest.resize(global_recv_data->num_msgs);
    vals.resize(global_recv_data->num_msgs, mpil_comm->rank_node);
    for (int i = 0; i < global_recv_data->num_msgs; i++)
        dest[i] = get_global_proc(mpil_comm, global_recv_data->procs[i], local_rank);
    MPIL_Alltoall_crs(global_recv_data->num_msgs,
                      dest.data(),
                      1, 
                      MPI_INT,
                      vals.data(),
                      &n_recvs,
                      &src,
                      1,
                      MPI_INT,
                      (void**) &recvbuf,
                      mpil_info,
                      mpil_comm);
    for (int i = 0; i < n_recvs; i++)
        nodes[num_nodes + recvbuf[i]] = src[i];

    MPIL_Free(src);
    MPIL_Free(recvbuf);


    MPI_Allreduce(MPI_IN_PLACE,
                  nodes.data(),
                  2*num_nodes,
                  MPI_INT,
                  MPI_MAX,
                  mpil_comm->local_comm);

    for (int i = 0; i < global_send_data->num_msgs; i++)
        global_send_data->procs[i] = nodes[num_nodes + global_send_data->procs[i]];
    for (int i = 0; i < global_recv_data->num_msgs; i++)
        global_recv_data->procs[i] = nodes[global_recv_data->procs[i]];

    MPIL_Info_free(&mpil_info);

}

void form_global_map(const CommData* map_data, std::map<long, int>& global_map)
{
    int idx;

    for (int i = 0; i < map_data->size_msgs; i++)
    {
        idx             = map_data->indices[i];
        global_map[idx] = i;
    }
}
void map_indices(CommData* idx_data, std::map<long, int>& global_map)
{
    int idx;

    for (int i = 0; i < idx_data->size_msgs; i++)
    {
        idx                  = idx_data->indices[i];
        idx_data->indices[i] = global_map[idx];
    }
}

void map_indices(CommData* idx_data, const CommData* map_data)
{
    std::map<long, int> global_map;
    form_global_map(map_data, global_map);
    map_indices(idx_data, global_map);
}

void remove_duplicates(CommData* comm_pkg)
{
    int start, end;
    int has_data = comm_pkg->size_msgs;

    for (int i = 0; i < comm_pkg->num_msgs; i++)
    {
        start = comm_pkg->indptr[i];
        end   = comm_pkg->indptr[i + 1];
        if (has_data)
        {
            std::sort(comm_pkg->indices + start, comm_pkg->indices + end);
        }
    }

    comm_pkg->size_msgs = 0;
    start               = comm_pkg->indptr[0];
    for (int i = 0; i < comm_pkg->num_msgs; i++)
    {
        end = comm_pkg->indptr[i + 1];
        if (has_data)
        {
            comm_pkg->indices[comm_pkg->size_msgs++] = comm_pkg->indices[start];
            for (int j = start; j < end - 1; j++)
            {
                if (comm_pkg->indices[j + 1] != comm_pkg->indices[j])
                {
                    comm_pkg->indices[comm_pkg->size_msgs++] = comm_pkg->indices[j + 1];
                }
            }
        }
        start                   = end;
        comm_pkg->indptr[i + 1] = comm_pkg->size_msgs;
        comm_pkg->counts[i]     = comm_pkg->indptr[i+1] - comm_pkg->indptr[i];
    }
}



void init_num_msgs(CommData* data, int num_msgs)
{
    data->num_msgs = num_msgs;
    if (data->num_msgs)
    {
        data->procs = (int*)malloc(sizeof(int) * data->num_msgs);
        data->counts = (int*)malloc(sizeof(int) * data->num_msgs);
    }
    data->indptr    = (int*)malloc(sizeof(int) * (data->num_msgs + 1));
    data->indptr[0] = 0;
}

void init_size_msgs(CommData* data, int size_msgs)
{
    data->size_msgs = size_msgs;
    if (data->size_msgs)
    {
        data->indices = (int*)malloc(data->size_msgs * sizeof(int));
    }
}

void destroy_comm_data(CommData* data)
{
    if (data->procs)
    {
        free(data->procs);
    }
    if (data->indptr)
    {
        free(data->indptr);
    }
    if (data->counts)
    {
        free(data->counts);
    }
    if (data->indices)
    {
        free(data->indices);
    }
    if (data->buffer)
    {
        free(data->buffer);
    }

    free(data);
}
