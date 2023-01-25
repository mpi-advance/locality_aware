#ifndef MPI_ADVANCE_TEST_NEIGHBOR_DATA_HPP
#define MPI_ADVANCE_TEST_NEIGHBOR_DATA_HPP

template <typename U>
struct MPIX_Data
{
    int num_msgs;
    int size_msgs; 
    std::vector<int> procs;
    std::vector<int> counts;
    std::vector<U> indptr;
    std::vector<int> indices;
    std::vector<MPI_Request> requests;
    std::vector<int> buffer;
};

// Form random communication
template <typename U>
void form_initial_communicator(int local_size, MPIX_Data<U>* send_data, MPIX_Data<U>* recv_data)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_n = 2;
    int max_s = 2;
    // int max_s = local_size;

    // Declare Variables
    srand(49352034 + rank);
    int n_sends = (rand() % max_n) + 1; // Between 1 and max_n msgs sent
    int first_idx = local_size * rank;
    int last_idx = local_size * (rank + 1) - 1;
    int tag = 4935;
    int start, end, proc;
    int size, ctr;
    std::vector<int> comm_procs(num_procs, 0);
    MPI_Status recv_status;
    
    // Create standard communication
    // Random send procs / data
    for (int i = 0; i < n_sends; i++)
    {
        proc = rand() % num_procs;
        while (proc == rank || comm_procs[proc] == 1)
        {
            proc = rand() % num_procs;    
        }
        comm_procs[proc] = 1;
    }
    for (int i = 0; i < num_procs; i++)
    {
        if (comm_procs[i])
        {
            send_data->procs.push_back(i);
        }
    }
    send_data->num_msgs = send_data->procs.size();
    send_data->counts.resize(send_data->num_msgs);
    send_data->indptr.resize(send_data->num_msgs + 1);
    send_data->requests.resize(send_data->num_msgs);

    ctr = 0;
    send_data->indptr[0] = 0;
    for (int i = 0; i < send_data->num_msgs; i++)
    {
        size = (rand() % max_s) + 1;
        for (int j = 0; j < size; j++)
        {
            send_data->indices.push_back(ctr++);
            if (ctr >= local_size) ctr = 0;
        }
        send_data->indptr[i+1] = (U)(send_data->indices.size());
        send_data->counts[i] = (int)(send_data->indptr[i+1] - send_data->indptr[i]);
    }
    send_data->size_msgs = send_data->indices.size();
    send_data->buffer.resize(send_data->size_msgs);

    // Form recv_data (first gather number of messages sent to rank)
    MPI_Allreduce(MPI_IN_PLACE, comm_procs.data(), num_procs, MPI_INT,
            MPI_SUM, MPI_COMM_WORLD);
    recv_data->num_msgs = comm_procs[rank];
    recv_data->procs.resize(recv_data->num_msgs);
    recv_data->counts.resize(recv_data->num_msgs);
    recv_data->indptr.resize(recv_data->num_msgs + 1);
    recv_data->requests.resize(recv_data->num_msgs);

    for (int i = 0; i < send_data->num_msgs; i++)
    {
        proc = send_data->procs[i];
        start = (int)send_data->indptr[i];
        end = (int)send_data->indptr[i+1];
        send_data->buffer[i] = end - start;
        MPI_Isend(&(send_data->buffer[i]), 1, MPI_INT, proc, tag, 
                MPI_COMM_WORLD, &send_data->requests[i]);
    }

    recv_data->indptr[0] = 0;
    for (int i = 0; i < recv_data->num_msgs; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &recv_status);
        proc = recv_status.MPI_SOURCE;
        MPI_Recv(&size, 1, MPI_INT, proc, tag, MPI_COMM_WORLD, &recv_status);
        recv_data->procs[i] = proc;
        recv_data->indptr[i+1] = recv_data->indptr[i] + (U)(size);
        recv_data->counts[i] = size;
    }
    recv_data->size_msgs = (int)(recv_data->indptr[recv_data->num_msgs]);
    recv_data->buffer.resize(recv_data->size_msgs);

    if (send_data->num_msgs)
    {
        MPI_Waitall(send_data->num_msgs, send_data->requests.data(),
                MPI_STATUSES_IGNORE);
    }
}


template <typename U>
void form_global_indices(int local_size, MPIX_Data<U> send_data, MPIX_Data<U> recv_data,
        std::vector<long>& global_send_idx, std::vector<long>& global_recv_idx)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int first = local_size * rank;
    int tag = 29354;
    for (int i = 0; i < send_data.num_msgs; i++)
    {
        int proc = send_data.procs[i];
        U start = send_data.indptr[i];
        U end = send_data.indptr[i+1];
        for (U j = start; j < end; j++)
        {
            int idx = send_data.indices[j];
            global_send_idx[j] = first + idx;
        }
        MPI_Isend(&(global_send_idx[start]), (int)(end - start), MPI_LONG, proc,
                tag, MPI_COMM_WORLD, &(send_data.requests[i]));
    }
    for (int i = 0; i < recv_data.num_msgs; i++)
    {
        int proc = recv_data.procs[i];
        U start = recv_data.indptr[i];
        U end = recv_data.indptr[i+1];
        MPI_Irecv(&(global_recv_idx[start]), (int)(end - start), MPI_LONG, proc,
                tag, MPI_COMM_WORLD, &(recv_data.requests[i]));
    }
    if (send_data.num_msgs) MPI_Waitall(send_data.num_msgs, 
            send_data.requests.data(), MPI_STATUSES_IGNORE);
    if (recv_data.num_msgs) MPI_Waitall(recv_data.num_msgs, 
            recv_data.requests.data(), MPI_STATUSES_IGNORE);
}


#endif
