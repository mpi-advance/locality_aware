#include "neighbor.h"
#include "neighbor_persistent.h"
#include "persistent/persistent.h"
#include <vector>
#include <algorithm>
#include <map>

/******************************************
 ****
 **** Helper Methods
 ****
 ******************************************/

void map_procs_to_nodes(LocalityComm* locality, const int orig_num_msgs,
        const int* orig_procs, const int* orig_counts,
        std::vector<int>& msg_nodes, std::vector<int>& msg_node_to_local,
        bool incr);
void form_local_comm(const int orig_num_sends, const int* orig_send_procs,
        const int* orig_send_ptr, const int* orig_sendcounts, const long* orig_send_indices,
        const std::vector<int>& nodes_to_local, CommData* send_data,
        CommData* recv_data, CommData* local_data,
        std::vector<int>& recv_idx_nodes,
        LocalityComm* locality, const int tag);
void form_global_comm(CommData* local_data, CommData* global_data,
        std::vector<int>& local_data_nodes, const MPIX_Comm* mpix_comm, int tag);
void update_global_comm(LocalityComm* locality);
void form_global_map(const CommData* map_data, std::map<long, int>& global_map);
void map_indices(CommData* idx_data, std::map<long, int>& global_map);
void map_indices(CommData* idx_data, const CommData* map_data);
void remove_duplicates(CommData* comm_pkg);
void remove_duplicates(CommPkg* data);
void remove_duplicates(LocalityComm* locality);
void update_indices(LocalityComm* locality, 
        std::map<long, int>& send_global_to_local,
        std::map<long, int>& recv_global_to_local);


/******************************************
 ****
 **** Main Methods
 ****
 ******************************************/

// Initialize NAPComm* structure, to be used for any number of
// instances of communication
void init_locality(const int n_sends, 
        const int* send_procs, 
        const int* send_indptr,
        const int* sendcounts,
        const int n_recvs,
        const int* recv_procs,
        const int* recv_indptr,
        const int* recvcounts,
        const long* global_send_indices,
        const long* global_recv_indices,
        const MPI_Datatype sendtype, 
        const MPI_Datatype recvtype,
        MPIX_Comm* mpix_comm,
        MPIX_Request* request)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(mpix_comm->global_comm, &rank);
    MPI_Comm_size(mpix_comm->global_comm, &num_procs);

    // Initialize structure
    LocalityComm* locality_comm;
    init_locality_comm(&locality_comm, mpix_comm, sendtype, recvtype);

    // Find global send nodes
    std::vector<int> send_nodes;
    std::vector<int> send_node_to_local;
    map_procs_to_nodes(locality_comm, 
            n_sends, 
            send_procs, 
            sendcounts,
            send_nodes, 
            send_node_to_local, 
            true);

    // Form initial send local comm
    std::vector<int> recv_idx_nodes;
    form_local_comm(n_sends, 
            send_procs, 
            send_indptr, 
            sendcounts,
            global_send_indices, 
            send_node_to_local,
            locality_comm->local_S_comm->send_data, 
            locality_comm->local_S_comm->recv_data,
            locality_comm->local_L_comm->send_data, 
            recv_idx_nodes,
            locality_comm, 
            19483);

    // Form global send data
    form_global_comm(locality_comm->local_S_comm->recv_data, 
            locality_comm->global_comm->send_data,
            recv_idx_nodes, 
            mpix_comm, 
            93284);

    // Find global recv nodes
    std::vector<int> recv_nodes;
    std::vector<int> recv_node_to_local;
    map_procs_to_nodes(locality_comm, 
            n_recvs, 
            recv_procs, 
            recvcounts,
            recv_nodes, 
            recv_node_to_local, 
            false);

    // Form final recv local comm
    std::vector<int> send_idx_nodes;
    form_local_comm(n_recvs,
            recv_procs,
            recv_indptr, 
            recvcounts,
            global_recv_indices,
            recv_node_to_local,
            locality_comm->local_R_comm->recv_data, 
            locality_comm->local_R_comm->send_data,
            locality_comm->local_L_comm->recv_data, 
            send_idx_nodes, 
            locality_comm,
            32048);

    // Form global recv data
    form_global_comm(locality_comm->local_R_comm->send_data,
            locality_comm->global_comm->recv_data,
            send_idx_nodes,
            locality_comm->communicators,
            93284);

    // Update procs for global_comm send and recvs
    update_global_comm(locality_comm);

    // Update send and receive indices
    std::map<long, int> send_global_to_local;
    std::map<long, int> recv_global_to_local;
    int ctr = 0;
    int start, end;
    for (int i = 0; i < n_sends; i++)
    {
        start = send_indptr[i];
        end = start + sendcounts[i];
        for (int j = start; j < end; j++)
            send_global_to_local[global_send_indices[ctr++]] = j;
    }

    ctr = 0;
    for (int i = 0; i < n_recvs; i++)
    {
        start = recv_indptr[i];
        end = start + recvcounts[i];
        for (int j = start; j < end; j++)
            recv_global_to_local[global_recv_indices[ctr++]] = j;
    }

    update_indices(locality_comm, 
            send_global_to_local, 
            recv_global_to_local);

    // Initialize final variable (MPI_Request arrays, etc.)
    finalize_locality_comm(locality_comm);

    // Copy to pointer for return
    request->locality = locality_comm;
    request->tag = locality_comm->global_comm->tag;
}

// Destroy NAPComm* structure
void destroy_locality(MPIX_Request* request)
{
    destroy_locality_comm(request->locality);
}



/******************************************
 ****
 **** Helper Methods
 ****
 ******************************************/
// Map original communication processes to nodes on which they lie
// And assign local processes to each node
void map_procs_to_nodes(LocalityComm* locality, const int orig_num_msgs,
        const int* orig_procs, const int* orig_counts,
        std::vector<int>& msg_nodes, std::vector<int>& msg_node_to_local,
        bool incr)
{
    int rank, num_procs;
    int local_rank, local_num_procs;

    MPI_Comm_rank(locality->communicators->global_comm, &rank);
    MPI_Comm_size(locality->communicators->global_comm, &num_procs);
    MPI_Comm_rank(locality->communicators->local_comm, &local_rank);
    MPI_Comm_size(locality->communicators->local_comm, &local_num_procs);

    int proc, size, node;
    int local_proc;
    int inc;
    std::vector<int> node_sizes;

    int num_nodes = locality->communicators->num_nodes;
    int rank_node = locality->communicators->rank_node;

    // Map local msg_procs to local msg_nodes
    node_sizes.resize(num_nodes, 0);
    for (int i = 0; i < orig_num_msgs; i++)
    {
        proc = orig_procs[i];
        size = orig_counts[i];
        node = get_node(locality->communicators, proc);
        node_sizes[node] += size;
    }

    // Gather all send nodes and sizes among ranks local to node
    MPI_Allreduce(MPI_IN_PLACE, node_sizes.data(), num_nodes, MPI_INT, MPI_SUM, locality->communicators->local_comm);
    for (int i = 0; i < num_nodes; i++)
    {
        if (node_sizes[i] && i != rank_node)
        {
            msg_nodes.push_back(i);
        }
    }
    std::sort(msg_nodes.begin(), msg_nodes.end(),
            [&](const int i, const int j)
            {
                return node_sizes[i] > node_sizes[j];
            });

    // Map send_nodes to local ranks
    msg_node_to_local.resize(num_nodes, -1);
    if (incr)
    {
        local_proc = 0;
        inc = 1;
    }
    else
    {
        local_proc = local_num_procs - 1;
        inc = -1;
    }
    for (size_t i = 0; i < msg_nodes.size(); i++)
    {
        node = msg_nodes[i];
        msg_node_to_local[node] = local_proc;

        if (local_proc == local_num_procs - 1 && inc == 1)
            inc = -1;
        else if (local_proc == 0 && inc == -1)
           inc = 1;
        else
            local_proc += inc;
    }
}

// Form step of local communication (either initial local_S communicator
// or final local_L communicator) along with the corresponding portion
// of the fully local (local_L) communicator.
void form_local_comm(const int orig_num_sends, const int* orig_send_procs,
        const int* orig_send_ptr, const int* orig_sendcounts, const long* orig_send_indices,
        const std::vector<int>& nodes_to_local, CommData* send_data,
        CommData* recv_data, CommData* local_data,
        std::vector<int>& recv_idx_nodes,
        LocalityComm* locality, const int tag)
{
    // MPI_Information
    int rank, num_procs;
    int local_rank, local_num_procs;
    MPI_Comm_rank(locality->communicators->global_comm, &rank);
    MPI_Comm_size(locality->communicators->global_comm, &num_procs);
    MPI_Comm_rank(locality->communicators->local_comm, &local_rank);
    MPI_Comm_size(locality->communicators->local_comm, &local_num_procs);

    // Declare variables
    int global_proc, local_proc;
    int size, ctr, start_ctr;
    int start, end, node;
    int idx, proc_idx;
    int proc, global_idx;
    MPI_Status recv_status;

    std::vector<int> send_buffer;
    std::vector<MPI_Request> send_requests;
    std::vector<int> send_sizes;
    std::vector<int> recv_buffer;

    std::vector<int> orig_to_node;
    std::vector<int> local_idx;

    // Initialize variables
    orig_to_node.resize(orig_num_sends);
    local_idx.resize(local_num_procs);
    send_sizes.resize(local_num_procs, 0);

    // Allocate sizes
    init_num_msgs(send_data, local_num_procs);
    init_num_msgs(recv_data, local_num_procs);
    init_num_msgs(local_data, local_num_procs);

    // Form local_S_comm
    send_data->num_msgs = 0;
    local_data->num_msgs = 0;
    recv_data->num_msgs = 0;
    for (int i = 0; i < orig_num_sends; i++)
    {
        global_proc = orig_send_procs[i];
        size = orig_sendcounts[i];
        node = get_node(locality->communicators, global_proc);
        if (locality->communicators->rank_node != node)
        {
            local_proc = nodes_to_local[node];
            if (send_sizes[local_proc] == 0)
            {
                local_idx[local_proc] = send_data->num_msgs;
                send_data->procs[send_data->num_msgs++] = local_proc;
            }
            orig_to_node[i] = node;
            send_sizes[local_proc] += size;
        }
        else
        {
            orig_to_node[i] = -1;
            local_data->procs[local_data->num_msgs] = get_local_proc(locality->communicators, global_proc);
            local_data->size_msgs += size;
            local_data->num_msgs++;
            local_data->indptr[local_data->num_msgs] = local_data->size_msgs;
        }
    }
    init_size_msgs(local_data, local_data->size_msgs);

    for (int i = 0; i < send_data->num_msgs; i++)
    {
        local_proc = send_data->procs[i];
        send_data->indptr[i+1] = send_data->indptr[i] + send_sizes[local_proc];
        send_sizes[local_proc] = 0;
    }
    send_data->size_msgs = send_data->indptr[send_data->num_msgs];

    // Allocate send_indices and fill vector
    init_size_msgs(send_data, send_data->size_msgs);

    std::vector<int> send_idx_node(send_data->size_msgs);
    local_data->size_msgs = 0;
    ctr = 0;
    for (int i = 0; i < orig_num_sends; i++)
    {
        node = orig_to_node[i];
        start = orig_send_ptr[i];
        end = orig_send_ptr[i] + orig_sendcounts[i];
        if (node == -1)
        {
            for (int j = start; j < end; j++)
            {
                global_idx = orig_send_indices[ctr++];
                local_data->indices[local_data->size_msgs++] = global_idx;
            }
        }
        else
        {
            local_proc = nodes_to_local[node];
            proc_idx = local_idx[local_proc];
            for (int j = start; j < end; j++)
            {
                global_idx = orig_send_indices[ctr++];
                idx = send_data->indptr[proc_idx] + send_sizes[local_proc]++;
                send_data->indices[idx] = global_idx;
                send_idx_node[idx] = node;
            }
        }
    }

    // Send 'local_S_comm send' info (to form local_S recv)
    MPI_Allreduce(MPI_IN_PLACE, send_sizes.data(), local_num_procs,
            MPI_INT, MPI_SUM, locality->communicators->local_comm);
    recv_data->size_msgs = send_sizes[local_rank];
    init_size_msgs(recv_data, recv_data->size_msgs);
    recv_idx_nodes.resize(recv_data->size_msgs);

    send_buffer.resize(2*send_data->size_msgs);
    send_requests.resize(send_data->num_msgs);
    ctr = 0;
    start_ctr = 0;
    for (int i = 0; i < send_data->num_msgs; i++)
    {
        proc = send_data->procs[i];
        start = send_data->indptr[i];
        end = send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            send_buffer[ctr++] = send_data->indices[j];
            send_buffer[ctr++] = send_idx_node[j];
        }
        MPI_Isend(&send_buffer[start_ctr], ctr - start_ctr ,
                MPI_INT, proc, tag, locality->communicators->local_comm, &send_requests[i]);
        start_ctr = ctr;
    }


    std::vector<int> proc_pos(local_num_procs, -1);
    std::vector<int> recv_idx(recv_data->size_msgs);
    std::vector<int> tmpnodes(recv_data->size_msgs);
    std::vector<int> recvptr(local_num_procs+1);
    recvptr[0] = 0;
    ctr = 0;
    while (ctr < recv_data->size_msgs)
    {
        MPI_Probe(MPI_ANY_SOURCE, tag, locality->communicators->local_comm, &recv_status);
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_INT, &size);
        if (size > (int) recv_buffer.size())
            recv_buffer.resize(size);
        MPI_Recv(recv_buffer.data(), size, MPI_INT, proc, tag, locality->communicators->local_comm, &recv_status);
        proc_pos[proc] = recv_data->num_msgs;
        for (int i = 0; i < size; i += 2)
        {
            recv_idx[ctr] = recv_buffer[i];
            tmpnodes[ctr++] = recv_buffer[i+1];
        }
        recvptr[recv_data->num_msgs+1] = recvptr[recv_data->num_msgs] + (size/2);
        recv_data->num_msgs++;
    }

    // Reorder Recvs
    ctr = 0;
    int pos, old_start, new_start;
    for (int i = 0; i < local_num_procs; i++)
    {
        if (proc_pos[i] == -1) continue;

        recv_data->procs[ctr] = i;
        pos = proc_pos[i];
        old_start = recvptr[pos];
        new_start = recv_data->indptr[ctr];
        size = recvptr[pos+1] - old_start;
        recv_data->indptr[++ctr] = new_start + size;
        for (int j = 0; j < size; j++)
        {
            recv_data->indices[new_start+j] = recv_idx[old_start+j];
            recv_idx_nodes[new_start+j] = tmpnodes[old_start+j];
        }
    }


    if (send_data->num_msgs)
    {
        MPI_Waitall(send_data->num_msgs, send_requests.data(), MPI_STATUSES_IGNORE);
    }
}


// Form portion of inter-node communication (data corresponding to
// either global send or global recv), with node id currently in
// place of process with which to communicate
void form_global_comm(CommData* local_data, CommData* global_data,
        std::vector<int>& local_data_nodes, const MPIX_Comm* mpix_comm, int tag)
{
    std::vector<int> tmp_send_indices;
    std::vector<int> node_sizes;
    std::vector<int> node_ctr;

    // Get MPI Information
    int rank, num_procs;
    int local_rank, local_num_procs;
    MPI_Comm_rank(mpix_comm->global_comm, &rank);
    MPI_Comm_size(mpix_comm->global_comm, &num_procs);
    MPI_Comm_rank(mpix_comm->local_comm, &local_rank);
    MPI_Comm_size(mpix_comm->local_comm, &local_num_procs);
    int num_nodes = mpix_comm->num_nodes;

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
    global_data->num_msgs = 0;
    global_data->indptr[0] = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        if (node_sizes[i])
        {
            global_data->procs[global_data->num_msgs] = i;
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
        end = local_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            node = local_data_nodes[j];
            node_idx = node_sizes[node];
            idx = global_data->indptr[node_idx] + node_ctr[node_idx]++;
            global_data->indices[idx] = local_data->indices[j];
        }
    }
}

// Replace send and receive processes with the node id's currently in their place
void update_global_comm(LocalityComm* locality)
{
    int rank, num_procs;
    MPI_Comm_rank(locality->communicators->global_comm, &rank);
    MPI_Comm_size(locality->communicators->global_comm, &num_procs);
    int local_rank, local_num_procs;
    MPI_Comm_rank(locality->communicators->local_comm, &local_rank);
    MPI_Comm_size(locality->communicators->local_comm, &local_num_procs);
    int num_nodes = locality->communicators->num_nodes;

    int n_sends = locality->global_comm->send_data->num_msgs;
    int n_recvs = locality->global_comm->recv_data->num_msgs;
    int n_msgs = n_sends + n_recvs;
    MPI_Request* requests = NULL;
    int* send_buffer = NULL;
    int send_tag, recv_tag;
    MPIX_Comm_tag(locality->communicators, &send_tag);
    MPIX_Comm_tag(locality->communicators, &recv_tag);
    int node, global_proc;
    int num_to_recv;
    MPI_Status recv_status;
    std::vector<int> send_nodes(num_nodes, 0);
    std::vector<int> recv_nodes(num_nodes, 0);
    if (n_msgs)
    {
        requests = new MPI_Request[n_msgs];
        send_buffer = new int[n_msgs];
    }

    std::vector<int> comm_procs(num_procs, 0);
    for (int i = 0; i < n_sends; i++)
    {
        node = locality->global_comm->send_data->procs[i];
        global_proc = get_global_proc(locality->communicators, node, local_rank);
        comm_procs[global_proc]++;
        send_buffer[i] = locality->communicators->rank_node;
        MPI_Isend(&(send_buffer[i]), 1, MPI_INT, global_proc, send_tag,
                locality->communicators->global_comm, &(requests[i]));
    }
    MPI_Allreduce(MPI_IN_PLACE, comm_procs.data(), num_procs, MPI_INT,
            MPI_SUM, locality->communicators->global_comm);
    num_to_recv = comm_procs[rank];
    for (int i = 0; i < num_procs; i++)
        comm_procs[i] = 0;
    for (int i = 0; i < num_to_recv; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, send_tag, locality->communicators->global_comm, &recv_status);
        global_proc = recv_status.MPI_SOURCE;
        MPI_Recv(&node, 1, MPI_INT, global_proc, send_tag, locality->communicators->global_comm, &recv_status);
        recv_nodes[node] = global_proc;
    }
    
    for (int i = 0; i < n_recvs; i++)
    {
        node = locality->global_comm->recv_data->procs[i];
        global_proc = get_global_proc(locality->communicators, node, local_rank);
        comm_procs[global_proc]++;
        send_buffer[n_sends + i] = locality->communicators->rank_node;
        MPI_Isend(&(send_buffer[n_sends + i]), 1, MPI_INT, global_proc, recv_tag,
                locality->communicators->global_comm, &(requests[n_sends + i]));
    }
    MPI_Allreduce(MPI_IN_PLACE, comm_procs.data(), num_procs, MPI_INT,
            MPI_SUM, locality->communicators->global_comm);
    num_to_recv = comm_procs[rank];
    for (int i = 0; i < num_to_recv; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, recv_tag, locality->communicators->global_comm, &recv_status);
        global_proc = recv_status.MPI_SOURCE;
        MPI_Recv(&node, 1, MPI_INT, global_proc, recv_tag, locality->communicators->global_comm, &recv_status);
        send_nodes[node] = global_proc;
    }

    if (n_sends + n_recvs)
        MPI_Waitall(n_sends + n_recvs, requests, MPI_STATUSES_IGNORE);

    MPI_Allreduce(MPI_IN_PLACE, send_nodes.data(), num_nodes, MPI_INT, MPI_MAX, locality->communicators->local_comm);
    MPI_Allreduce(MPI_IN_PLACE, recv_nodes.data(), num_nodes, MPI_INT, MPI_MAX, locality->communicators->local_comm);

    for (int i = 0; i < n_sends; i++)
    {
        node = locality->global_comm->send_data->procs[i];
        locality->global_comm->send_data->procs[i] = send_nodes[node];
    }
    for (int i = 0; i < n_recvs; i++)
    {
        node = locality->global_comm->recv_data->procs[i];
        locality->global_comm->recv_data->procs[i] = recv_nodes[node];
    }

    if (requests) delete[] requests;
    if (send_buffer) delete[] send_buffer;
}

// Update indices:
// 1.) map initial sends to point to positions in original data
// 2.) map internal communication steps to point to correct
//     position in previously received data
// 3.) map final receives to points in original recv data
void form_global_map(const CommData* map_data, std::map<long, int>& global_map)
{
    int idx;

    for (int i = 0; i < map_data->size_msgs; i++)
    {
        idx = map_data->indices[i];
        global_map[idx] = i;
    }
}
void map_indices(CommData* idx_data, std::map<long, int>& global_map)
{
    int idx;

    for (int i = 0; i < idx_data->size_msgs; i++)
    {
        idx = idx_data->indices[i];
        idx_data->indices[i] = global_map[idx];
    }
}

void map_indices(CommData* idx_data, const CommData* map_data)
{
    std::map<long, int> global_map;
    form_global_map(map_data, global_map);
    map_indices(idx_data, global_map);
}

int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

void remove_duplicates(CommData* comm_pkg)
{
    int start, end;

    for (int i = 0; i < comm_pkg->num_msgs; i++)
    {
        start = comm_pkg->indptr[i];
        end = comm_pkg->indptr[i+1];
        std::sort(comm_pkg->indices+start, comm_pkg->indices+end);
    }

    comm_pkg->size_msgs = 0;
    start = comm_pkg->indptr[0];
    for (int i = 0; i < comm_pkg->num_msgs; i++)
    {
        end = comm_pkg->indptr[i+1];
        comm_pkg->indices[comm_pkg->size_msgs++] = comm_pkg->indices[start];
        for (int j  = start; j < end - 1; j++)
        {
            if (comm_pkg->indices[j+1] != comm_pkg->indices[j])
            {
                comm_pkg->indices[comm_pkg->size_msgs++] = comm_pkg->indices[j+1];
            }
        }
        start = end;
        comm_pkg->indptr[i+1] = comm_pkg->size_msgs;
    }
}

void remove_duplicates(CommPkg* data)
{
        remove_duplicates(data->send_data);
        remove_duplicates(data->recv_data);
}

void remove_duplicates(LocalityComm* locality)
{
    remove_duplicates(locality->local_S_comm);
    remove_duplicates(locality->local_R_comm);
    remove_duplicates(locality->global_comm);
}


void update_indices(LocalityComm* locality, 
        std::map<long, int>& send_global_to_local,
        std::map<long, int>& recv_global_to_local)
{
    // Remove duplicates
    remove_duplicates(locality);

    // Map global indices to usable indices
    map_indices(locality->global_comm->send_data, locality->local_S_comm->recv_data);
    map_indices(locality->local_R_comm->send_data, locality->global_comm->recv_data);
    map_indices(locality->local_S_comm->send_data, send_global_to_local);
    map_indices(locality->local_L_comm->send_data, send_global_to_local);
    map_indices(locality->local_R_comm->recv_data, recv_global_to_local);
    map_indices(locality->local_L_comm->recv_data, recv_global_to_local);

    // Don't need local_S or global recv indices (just contiguous)
    if (locality->local_S_comm->recv_data->indices)
    {
        free(locality->local_S_comm->recv_data->indices);
        locality->local_S_comm->recv_data->indices = NULL;
    }
    if (locality->global_comm->recv_data->indices)
    {
        free(locality->global_comm->recv_data->indices);
        locality->global_comm->recv_data->indices = NULL;
    }
}

