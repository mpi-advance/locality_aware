#include "nap_communicate.hpp"


/******************************************
 ****
 **** Main Methods
 ****
 ******************************************/

// Node-Aware Version of Isend
void MPIX_INAPsend(const void* buf, void* nap_comm_ptr,
        MPI_Datatype datatype, int tag,
        MPI_Comm comm)
{
    NAPComm* nap_comm = (NAPComm*) nap_comm_ptr;
    NAPData* nap_data = nap_comm->nap_data;

    NAPCommData* nap_send_data = new NAPCommData();
    nap_send_data->datatype = datatype;
    nap_send_data->tag = tag;

    int local_S_tag = tag + 1;
    int local_L_tag = tag + 3;
    int size, type_size, ctr;

    char* local_L_recv_data = NULL;
    char* local_S_recv_data = NULL;
    char* global_send_buffer = NULL;
    char* L_send_buffer = NULL;
    MPI_Request* send_requests = nap_comm->send_requests;
    MPI_Request* recv_requests = nap_comm->recv_requests;
    MPI_Request* L_send_requests = NULL;

    MPI_Type_size(datatype, &type_size);

    // Initial intra-node redistribution (step 1 in nap comm)
    MPIX_step_comm(nap_comm->local_S_comm, buf, &local_S_recv_data,
            local_S_tag, nap_comm->topo_info->local_comm, datatype, datatype,
            send_requests, recv_requests);

    if (nap_comm->local_S_comm->recv_data->size_msgs)
    {
        // Unpack previous recv into void*
        char* unpacked_buf = MPIX_NAP_unpack(local_S_recv_data, 
                nap_comm->local_S_comm->recv_data->size_msgs,
                datatype, comm);

        // Initialize Isends for inter-node step (step 2 in nap comm)
        MPIX_step_send(nap_comm->global_comm, local_S_recv_data, tag,
                comm, datatype, send_requests, &global_send_buffer);

        // Free void* buffer (packed into char* buf)
        delete[] unpacked_buf;
    }

    if (nap_comm->local_L_comm->send_data->num_msgs)
    {
        L_send_requests = &(send_requests[nap_comm->global_comm->send_data->num_msgs]);
        MPIX_step_send(nap_comm->local_L_comm, buf, local_L_tag, 
                nap_comm->topo_info->local_comm, datatype, L_send_requests, &L_send_buffer);
    }

    // Store global_send_requests and global_send_buffer, as to not free data
    // before sends are finished
    nap_send_data->global_buffer = global_send_buffer;
    nap_send_data->local_L_buffer = L_send_buffer;
    nap_data->send_data = nap_send_data;

    if (local_S_recv_data) delete[] local_S_recv_data;
}

// Node-Aware Version of Irecv
void MPIX_INAPrecv(void* buf, void* nap_comm_ptr,
        MPI_Datatype datatype, int tag,
        MPI_Comm comm)
{
    NAPComm* nap_comm = (NAPComm*) nap_comm_ptr;
    NAPData* nap_data = nap_comm->nap_data;

    NAPCommData* nap_recv_data = new NAPCommData();
    nap_data->mpi_comm = comm;
    nap_recv_data->tag = tag;
    nap_recv_data->buf = buf;
    nap_recv_data->datatype = datatype;
    MPI_Request* global_recv_requests = NULL;
    MPI_Request* recv_requests = nap_comm->recv_requests;
    MPI_Request* L_recv_requests = NULL;
    char* global_recv_buffer = NULL;
    char* local_L_buffer = NULL;

    int local_L_tag = tag + 3;

    // Initialize Irecvs for inter-node step (step 2 in nap comm)
    MPIX_step_recv(nap_comm->global_comm, tag, comm, datatype,
            recv_requests, &global_recv_buffer);

    if (nap_comm->local_L_comm->recv_data->num_msgs)
    {
        L_recv_requests = &(recv_requests[nap_comm->global_comm->recv_data->num_msgs]);
        MPIX_step_recv(nap_comm->local_L_comm, local_L_tag, nap_comm->topo_info->local_comm,
                datatype, L_recv_requests, &local_L_buffer);
    }

    // Store global_recv_requests and global_recv_buffer, as to not free data
    // before recvs are finished
    nap_recv_data->global_buffer = global_recv_buffer;
    nap_recv_data->local_L_buffer = local_L_buffer;
    nap_data->recv_data = nap_recv_data;
}

// Wait for Node-Aware Isends and Irecvs to complete
void MPIX_NAPwait(void* nap_comm_ptr)
{
    NAPComm* nap_comm = (NAPComm*) nap_comm_ptr;
    NAPData* nap_data = nap_comm->nap_data;

    NAPCommData* nap_send_data = nap_data->send_data;
    NAPCommData* nap_recv_data = nap_data->recv_data;
    MPI_Comm mpi_comm = nap_data->mpi_comm;

    char* local_R_recv_data = NULL;
    char* local_L_recv_data = nap_recv_data->local_L_buffer;
    char* global_send_buffer = nap_send_data->global_buffer;
    char* global_recv_buffer = nap_recv_data->global_buffer;
    MPI_Request* send_requests = nap_comm->send_requests;
    MPI_Request* recv_requests = nap_comm->recv_requests;
    MPI_Request* L_send_requests = NULL;
    MPI_Request* L_recv_requests = NULL;
    MPI_Datatype send_type = nap_send_data->datatype;
    MPI_Datatype recv_type = nap_recv_data->datatype;

    int local_R_tag = nap_recv_data->tag + 2;

    MPIX_step_waitall(nap_comm->global_comm, send_requests, recv_requests);
    if (nap_comm->local_L_comm->send_data->num_msgs)
        L_send_requests = &(send_requests[nap_comm->global_comm->send_data->num_msgs]);
    if (nap_comm->local_L_comm->recv_data->num_msgs)
        L_recv_requests = &(recv_requests[nap_comm->global_comm->recv_data->num_msgs]);
    MPIX_step_waitall(nap_comm->local_L_comm, L_send_requests, L_recv_requests);
    if (nap_send_data->local_L_buffer)
    {
        delete[] nap_send_data->local_L_buffer;
        nap_send_data->local_L_buffer = NULL;
    }

    // Map recv buffers from final intra node steps to correct locations in
    // recv_data
    local_L_recv_data = nap_recv_data->local_L_buffer;
    if (local_L_recv_data)
    {
        MPIX_intra_recv_map(nap_comm->local_L_comm, local_L_recv_data, nap_recv_data->buf,
                recv_type, nap_comm->topo_info->local_comm);
        delete[] local_L_recv_data;
        nap_recv_data->local_L_buffer = NULL;
    }

    // Final intra-node redistribution (step 3 in nap comm)
    // Unpack global_recv_buffer ot U[] (but don't know U here...)
    char* unpacked_buf = NULL;
    if (nap_comm->global_comm->recv_data->size_msgs)
    {
        // Unpack previous recv into void*
        unpacked_buf = MPIX_NAP_unpack(global_recv_buffer, 
                nap_comm->global_comm->recv_data->size_msgs,
                recv_type, mpi_comm);
    }

    // Initialize Isends for inter-node step (step 2 in nap comm)
    MPIX_step_comm(nap_comm->local_R_comm, global_recv_buffer, &local_R_recv_data,
            local_R_tag, nap_comm->topo_info->local_comm, recv_type, recv_type,
            send_requests, recv_requests);

    if (unpacked_buf)
        delete[] unpacked_buf;
    
    if (local_R_recv_data)
    {
        MPIX_intra_recv_map(nap_comm->local_R_comm, local_R_recv_data, nap_recv_data->buf,
                recv_type, nap_comm->topo_info->local_comm);
        delete[] local_R_recv_data;
    }
    nap_recv_data->buf = NULL;
    delete nap_send_data;
    delete nap_recv_data;
}




// Initialize NAPComm* structure, to be used for any number of
// instances of communication
void MPIX_NAPinit(const int n_sends, const int* send_procs, const int* send_indptr,
        const int* send_indices, const int n_recvs, const int* recv_procs,
        const int* recv_indptr, const int* global_send_indices,
        const int* global_recv_indices, const MPI_Comm mpi_comm,
        void** nap_comm_ptr)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &num_procs);

    // Create topo_data object
    topo_data* topology_info = new topo_data(mpi_comm);

    // Initialize structure
    NAPComm* nap_comm = new NAPComm(topology_info);

    // Find global send nodes
    std::vector<int> send_nodes;
    std::vector<int> send_node_to_local;
    map_procs_to_nodes(nap_comm, n_sends, send_procs, send_indptr,
            send_nodes, send_node_to_local, mpi_comm, true);

    // Form initial send local comm
    std::vector<int> recv_idx_nodes;
    form_local_comm(n_sends, send_procs, send_indptr, global_send_indices, send_node_to_local,
            nap_comm->local_S_comm->send_data, nap_comm->local_S_comm->recv_data,
            nap_comm->local_L_comm->send_data, recv_idx_nodes, mpi_comm,
            topology_info, 19483);

    // Form global send data
    form_global_comm(nap_comm->local_S_comm->recv_data, nap_comm->global_comm->send_data,
            recv_idx_nodes, mpi_comm, topology_info, 93284);

    // Find global recv nodes
    std::vector<int> recv_nodes;
    std::vector<int> recv_node_to_local;
    map_procs_to_nodes(nap_comm, n_recvs, recv_procs, recv_indptr,
            recv_nodes, recv_node_to_local, mpi_comm, false);

    // Form final recv local comm
    std::vector<int> send_idx_nodes;
    form_local_comm(n_recvs, recv_procs, recv_indptr, global_recv_indices, recv_node_to_local,
            nap_comm->local_R_comm->recv_data, nap_comm->local_R_comm->send_data,
            nap_comm->local_L_comm->recv_data, send_idx_nodes, mpi_comm,
            topology_info, 32048);

    // Form global recv data
    form_global_comm(nap_comm->local_R_comm->send_data, nap_comm->global_comm->recv_data,
            send_idx_nodes, mpi_comm, topology_info, 93284);

    // Update procs for global_comm send and recvs
    update_global_comm(nap_comm, topology_info, mpi_comm);

    // Update send and receive indices
    int send_idx_size = send_indptr[n_sends];
    int recv_idx_size = recv_indptr[n_recvs];
    std::map<int, int> send_global_to_local;
    std::map<int, int> recv_global_to_local;
    for (int i = 0; i < send_idx_size; i++)
        send_global_to_local[global_send_indices[i]] = send_indices[i];
    for (int i = 0; i < recv_idx_size; i++)
        recv_global_to_local[global_recv_indices[i]] = i;
    update_indices(nap_comm, send_global_to_local, recv_global_to_local);


    // Initialize final variable (MPI_Request arrays, etc.)
    nap_comm->finalize();

    // Copy to pointer for return
    *nap_comm_ptr = (void**) nap_comm;
}

// Destroy NAPComm* structure
void MPIX_NAPDestroy(void* nap_comm_ptr)
{
    NAPComm* nap_comm = (NAPComm*) nap_comm_ptr;
    topo_data* topo_info = nap_comm->topo_info;
    delete topo_info;
    delete nap_comm;
}



/******************************************
 ****
 **** Helper Methods
 ****
 ******************************************/
// Map original communication processes to nodes on which they lie
// And assign local processes to each node
void map_procs_to_nodes(NAPComm* nap_comm, const int orig_num_msgs,
        const int* orig_procs, const int* orig_indptr,
        std::vector<int>& msg_nodes, std::vector<int>& msg_node_to_local,
        MPI_Comm mpi_comm, bool incr)
{
    int rank, num_procs;
    int local_rank, local_num_procs;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &num_procs);
    MPI_Comm& local_comm = nap_comm->topo_info->local_comm;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_num_procs);

    int proc, size, node;
    int local_proc;
    int inc;
    std::vector<int> node_sizes;

    int num_nodes = nap_comm->topo_info->num_nodes;
    int rank_node = nap_comm->topo_info->rank_node;

    // Map local msg_procs to local msg_nodes
    node_sizes.resize(num_nodes, 0);
    for (int i = 0; i < orig_num_msgs; i++)
    {
        proc = orig_procs[i];
        size = orig_indptr[i+1] - orig_indptr[i];
        node = nap_comm->topo_info->get_node(proc);
        node_sizes[node] += size;
    }

    // Gather all send nodes and sizes among ranks local to node
    MPI_Allreduce(MPI_IN_PLACE, node_sizes.data(), num_nodes, MPI_INT, MPI_SUM, local_comm);
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
    for (int i = 0; i < msg_nodes.size(); i++)
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
        const int* orig_send_ptr, const int* orig_send_indices,
        const std::vector<int>& nodes_to_local, comm_data* send_data,
        comm_data* recv_data, comm_data* local_data,
        std::vector<int>& recv_idx_nodes, MPI_Comm mpi_comm,
        topo_data* topo_info, const int tag)
{
    // MPI_Information
    int rank, num_procs;
    int local_rank, local_num_procs;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &num_procs);
    MPI_Comm& local_comm = topo_info->local_comm;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_num_procs);

    // Declare variables
    int global_proc, local_proc;
    int size, ctr, start_ctr;
    int start, end, node;
    int idx, proc_idx;
    int proc;
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

    send_data->init_num_data(local_num_procs);
    recv_data->init_num_data(local_num_procs);
    local_data->init_num_data(local_num_procs);

    // Form local_S_comm
    for (int i = 0; i < orig_num_sends; i++)
    {
        global_proc = orig_send_procs[i];
        size = orig_send_ptr[i+1] - orig_send_ptr[i];
        node = topo_info->get_node(global_proc);
        if (topo_info->rank_node != node)
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
            local_data->procs[local_data->num_msgs] = topo_info->get_local_proc(global_proc);
            local_data->size_msgs += size;
            local_data->num_msgs++;
            local_data->indptr[local_data->num_msgs] = local_data->size_msgs;
        }
    }
    if (local_data->size_msgs)
        local_data->indices = new int[local_data->size_msgs];

    for (int i = 0; i < send_data->num_msgs; i++)
    {
        local_proc = send_data->procs[i];
        send_data->indptr[i+1] = send_data->indptr[i] + send_sizes[local_proc];
        send_sizes[local_proc] = 0;
    }
    send_data->size_msgs = send_data->indptr[send_data->num_msgs];

    // Allocate send_indices and fill vector
    if (send_data->size_msgs)
        send_data->indices = new int[send_data->size_msgs];
    std::vector<int> send_idx_node(send_data->size_msgs);
    local_data->size_msgs = 0;
    for (int i = 0; i < orig_num_sends; i++)
    {
        node = orig_to_node[i];
        start = orig_send_ptr[i];
        end = orig_send_ptr[i+1];
        if (node == -1)
        {
            for (int j = start; j < end; j++)
            {
                local_data->indices[local_data->size_msgs++] = orig_send_indices[j];
            }
        }
        else
        {
            local_proc = nodes_to_local[node];
            proc_idx = local_idx[local_proc];
            for (int j = start; j < end; j++)
            {
                idx = send_data->indptr[proc_idx] + send_sizes[local_proc]++;
                send_data->indices[idx] = orig_send_indices[j];
                send_idx_node[idx] = node;
            }
        }
    }

    // Send 'local_S_comm send' info (to form local_S recv)
    MPI_Allreduce(MPI_IN_PLACE, send_sizes.data(), local_num_procs,
            MPI_INT, MPI_SUM, local_comm);
    recv_data->size_msgs = send_sizes[local_rank];
    if (recv_data->size_msgs)
        recv_data->indices = new int[recv_data->size_msgs];
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
                MPI_INT, proc, tag, local_comm, &send_requests[i]);
        start_ctr = ctr;
    }

    ctr = 0;
    while (ctr < recv_data->size_msgs)
    {
        MPI_Probe(MPI_ANY_SOURCE, tag, local_comm, &recv_status);
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_INT, &size);
        if (size > recv_buffer.size())
            recv_buffer.resize(size);
        MPI_Recv(recv_buffer.data(), size, MPI_INT, proc, tag, local_comm, &recv_status);
        for (int i = 0; i < size; i += 2)
        {
            recv_data->indices[ctr] = recv_buffer[i];
            recv_idx_nodes[ctr++] = recv_buffer[i+1];
        }
        recv_data->procs[recv_data->num_msgs] = proc;
        recv_data->indptr[recv_data->num_msgs + 1] = recv_data->indptr[recv_data->num_msgs] + (size / 2);
        recv_data->num_msgs++;
    }

    if (send_data->num_msgs)
    {
        MPI_Waitall(send_data->num_msgs, send_requests.data(), MPI_STATUSES_IGNORE);
    }
}

// Form portion of inter-node communication (data corresponding to
// either global send or global recv), with node id currently in
// place of process with which to communicate
void form_global_comm(comm_data* local_data, comm_data* global_data,
        std::vector<int>& local_data_nodes, MPI_Comm mpi_comm,
        topo_data* topo_info, int tag)
{
    std::vector<int> tmp_send_indices;
    std::vector<int> node_sizes;
    std::vector<int> node_ctr;

    // Get MPI Information
    int rank, num_procs;
    int local_rank, local_num_procs;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &num_procs);
    MPI_Comm& local_comm = topo_info->local_comm;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_num_procs);
    int num_nodes = topo_info->num_nodes;
    int rank_node = topo_info->rank_node;

    int node_idx;
    int start, end, idx;
    int ctr, node, size;

    node_sizes.resize(num_nodes, 0);

    for (int i = 0; i < local_data->num_msgs; i++)
    {
        node = local_data_nodes[i];
        size = local_data->indptr[i+1] - local_data->indptr[i];
        if (node_sizes[node] == 0)
        {
            global_data->num_msgs++;
        }
        node_sizes[node] += size;
    }
    if (global_data->num_msgs)
        global_data->procs = new int[global_data->num_msgs];
    global_data->indptr = new int[global_data->num_msgs+1];
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

    if (global_data->size_msgs)
        global_data->indices = new int[global_data->size_msgs];
    for (int i = 0; i < local_data->num_msgs; i++)
    {
        node = local_data_nodes[i];
        node_idx = node_sizes[node];
        start = local_data->indptr[i];
        end = local_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = global_data->indptr[node_idx] + node_ctr[node_idx]++;
            global_data->indices[idx] = local_data->indices[j];
        }
    }
}

// Replace send and receive processes with the node id's currently in their place
void update_global_comm(NAPComm* nap_comm, topo_data* topo_info, MPI_Comm mpi_comm)
{
    int rank, num_procs;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &num_procs);
    MPI_Comm& local_comm = topo_info->local_comm;
    int local_rank, local_num_procs;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_num_procs);
    int num_nodes = topo_info->num_nodes;

    int n_sends = nap_comm->global_comm->send_data->num_msgs;
    int n_recvs = nap_comm->global_comm->recv_data->num_msgs;
    int n_msgs = n_sends + n_recvs;
    MPI_Request* requests = NULL;
    int* send_buffer = NULL;
    int send_tag = 32148532;
    int recv_tag = 52395234;
    int node, global_proc, tag;
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
        node = nap_comm->global_comm->send_data->procs[i];
        global_proc = topo_info->get_global_proc(node, local_rank);
        comm_procs[global_proc]++;
        send_buffer[i] = nap_comm->topo_info->rank_node;
        MPI_Isend(&send_buffer[i], 1, MPI_INT, global_proc, send_tag,
                mpi_comm, &requests[i]);
    }
    for (int i = 0; i < n_recvs; i++)
    {
        node = nap_comm->global_comm->recv_data->procs[i];
        global_proc = topo_info->get_global_proc(node, local_rank);
        comm_procs[global_proc]++;
        send_buffer[n_sends + i] = nap_comm->topo_info->rank_node;
        MPI_Isend(&send_buffer[n_sends + i], 1, MPI_INT, global_proc, recv_tag,
                mpi_comm, &requests[n_sends + i]);
    }

    MPI_Allreduce(MPI_IN_PLACE, comm_procs.data(), num_procs, MPI_INT,
            MPI_SUM, MPI_COMM_WORLD);
    int num_to_recv = comm_procs[rank];

    for (int i = 0; i < num_to_recv; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, mpi_comm, &recv_status);
        global_proc = recv_status.MPI_SOURCE;
        tag = recv_status.MPI_TAG;
        MPI_Recv(&node, 1, MPI_INT, global_proc, tag, mpi_comm, &recv_status);
        if (tag == send_tag)
        {
            recv_nodes[node] = global_proc;
        }
        else
        {
            send_nodes[node] = global_proc;
        }
    }

    if (n_sends + n_recvs)
        MPI_Waitall(n_sends + n_recvs, requests, MPI_STATUSES_IGNORE);

    MPI_Allreduce(MPI_IN_PLACE, send_nodes.data(), num_nodes, MPI_INT, MPI_MAX, local_comm);
    MPI_Allreduce(MPI_IN_PLACE, recv_nodes.data(), num_nodes, MPI_INT, MPI_MAX, local_comm);

    for (int i = 0; i < n_sends; i++)
    {
        node = nap_comm->global_comm->send_data->procs[i];
        nap_comm->global_comm->send_data->procs[i] = send_nodes[node];
    }
    for (int i = 0; i < n_recvs; i++)
    {
        node = nap_comm->global_comm->recv_data->procs[i];
        nap_comm->global_comm->recv_data->procs[i] = recv_nodes[node];
    }

    delete[] requests;
    delete[] send_buffer;
}

// Update indices:
// 1.) map initial sends to point to positions in original data
// 2.) map internal communication steps to point to correct
//     position in previously received data
// 3.) map final receives to points in original recv data
void form_global_map(const comm_data* map_data, std::map<int, int>& global_map)
{
    int idx;

    for (int i = 0; i < map_data->size_msgs; i++)
    {
        idx = map_data->indices[i];
        global_map[idx] = i;
    }
}
void map_indices(comm_data* idx_data, std::map<int, int>& global_map)
{
    int idx;

    for (int i = 0; i < idx_data->size_msgs; i++)
    {
        idx = idx_data->indices[i];
        idx_data->indices[i] = global_map[idx];
    }
}
void map_indices(comm_data* idx_data, const comm_data* map_data)
{
    std::map<int, int> global_map;
    form_global_map(map_data, global_map);
    map_indices(idx_data, global_map);
}
void update_indices(NAPComm* nap_comm, std::map<int, int>& send_global_to_local,
        std::map<int, int>& recv_global_to_local)
{
    // Remove duplicates
    nap_comm->global_comm->send_data->remove_duplicates();
    nap_comm->global_comm->recv_data->remove_duplicates();
    nap_comm->local_S_comm->send_data->remove_duplicates();
    nap_comm->local_S_comm->recv_data->remove_duplicates();
    nap_comm->local_R_comm->send_data->remove_duplicates();
    nap_comm->local_R_comm->recv_data->remove_duplicates();

    // Map global indices to usable indices
    map_indices(nap_comm->global_comm->send_data, nap_comm->local_S_comm->recv_data);
    map_indices(nap_comm->local_R_comm->send_data, nap_comm->global_comm->recv_data);
    map_indices(nap_comm->local_S_comm->send_data, send_global_to_local);
    map_indices(nap_comm->local_L_comm->send_data, send_global_to_local);
    map_indices(nap_comm->local_R_comm->recv_data, recv_global_to_local);
    map_indices(nap_comm->local_L_comm->recv_data, recv_global_to_local);

    // Don't need local_S or global recv indices (just contiguous)
    delete[] nap_comm->local_S_comm->recv_data->indices;
    delete[] nap_comm->global_comm->recv_data->indices;
    nap_comm->local_S_comm->recv_data->indices = NULL;
    nap_comm->global_comm->recv_data->indices = NULL;
}




char* MPIX_NAP_unpack(char* packed_buf, int size, MPI_Datatype datatype, MPI_Comm comm)
{
    int type_size, pack_size;
    int ctr = 0;

    MPI_Type_size(datatype, &type_size);
    char* unpacked_buf = new char[size*type_size];

    MPI_Pack_size(size, datatype, comm, &pack_size);
    MPI_Unpack(packed_buf, pack_size, &ctr, unpacked_buf, size, datatype, comm);

    return unpacked_buf;
}


// Intra/Inter-Node Waitall
void MPIX_step_waitall(comm_pkg* comm, MPI_Request* send_requests,
        MPI_Request* recv_requests)
{
    int flag;
    if (comm->send_data->num_msgs)
    {
        MPI_Waitall(comm->send_data->num_msgs, send_requests, MPI_STATUSES_IGNORE);
    }
    if (comm->recv_data->num_msgs)
    {
        MPI_Waitall(comm->recv_data->num_msgs, recv_requests, MPI_STATUSES_IGNORE);
    }
}


// Intra-Node Communication
void MPIX_step_comm(comm_pkg* comm, const void* send_data, char** recv_data,
        int tag, MPI_Comm local_comm, MPI_Datatype send_type, MPI_Datatype recv_type,
        MPI_Request* send_requests, MPI_Request* recv_requests)
{
    char* send_buffer = NULL;
    char* recv_buffer = NULL;

    MPIX_step_send(comm, send_data, tag, local_comm, send_type,
            send_requests, &send_buffer);
    MPIX_step_recv(comm, tag, local_comm, recv_type, recv_requests,
            &recv_buffer);
    MPIX_step_waitall(comm, send_requests, recv_requests); 

    if (send_buffer) delete[] send_buffer;
    if (recv_buffer) *recv_data = recv_buffer;
}

// Inter-node Isend
void MPIX_step_send(comm_pkg* comm, const void* send_data,
        int tag, MPI_Comm mpi_comm, MPI_Datatype mpi_type,
        MPI_Request* send_requests, char** send_buffer_ptr)
{
    int idx, proc, start, end;
    int size, type_size, addr;
    char* send_buffer = NULL;
    const char* data = reinterpret_cast<const char*>(send_data);
    MPI_Type_size(mpi_type, &type_size);
    if (comm->send_data->size_msgs)
    {
        MPI_Pack_size(comm->send_data->size_msgs, mpi_type, mpi_comm, &size);
        send_buffer = new char[size];
    }

    int ctr = 0;
    int prev_ctr = 0;
    for (int i = 0; i < comm->send_data->num_msgs; i++)
    {
        proc = comm->send_data->procs[i];
        start = comm->send_data->indptr[i];
        end = comm->send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = comm->send_data->indices[j];
            MPI_Pack(&(data[idx*type_size]), 1, mpi_type, 
                    send_buffer, size, &ctr, mpi_comm);
        }
        MPI_Isend(&send_buffer[prev_ctr], ctr - prev_ctr, MPI_PACKED, proc, tag,
                mpi_comm, &send_requests[i]);
        prev_ctr = ctr;
    }

    if (send_buffer) *send_buffer_ptr = send_buffer;
}

// Inter-Node Irecvs
void MPIX_step_recv(comm_pkg* comm,
        int tag, MPI_Comm mpi_comm, MPI_Datatype mpi_type,
        MPI_Request* recv_requests, char** recv_buffer_ptr)
{
    int proc, start, end, size;
    char* recv_buffer = NULL;
    if (comm->recv_data->size_msgs)
    {
        MPI_Pack_size(comm->recv_data->size_msgs, mpi_type, mpi_comm, &size);
        recv_buffer = new char[size];
    }

    int ctr = 0;
    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        proc = comm->recv_data->procs[i];
        start = comm->recv_data->indptr[i];
        end = comm->recv_data->indptr[i+1];
        MPI_Pack_size(end - start, mpi_type, mpi_comm, &size);
        MPI_Irecv(&recv_buffer[ctr], size, MPI_PACKED, proc, tag,
                mpi_comm, &recv_requests[i]);
        ctr += size;
    }

    if (recv_buffer) *recv_buffer_ptr = recv_buffer;
}

// Map received values to the appropriate locations
void MPIX_intra_recv_map(comm_pkg* comm, char* intra_recv_data, void* inter_recv_data,
        MPI_Datatype mpi_type, MPI_Comm mpi_comm)
{
    int idx, addr;
    int size, type_size;
    char* char_ptr = reinterpret_cast<char*>(inter_recv_data);
    int* int_ptr = reinterpret_cast<int*>(inter_recv_data);
    MPI_Type_size(mpi_type, &type_size);
    MPI_Pack_size(comm->recv_data->size_msgs, mpi_type, mpi_comm, &size);

    int ctr = 0;
    for (int i = 0; i < comm->recv_data->size_msgs; i++)
    {
        idx = comm->recv_data->indices[i];
        MPI_Unpack(intra_recv_data, size, &ctr, &(char_ptr[idx*type_size]), 
                1, mpi_type, mpi_comm);
    }
}

