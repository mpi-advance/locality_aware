// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails
// while EXPECT_* variants continue with the run.


#include "gtest/gtest.h"
#include "src/nap_comm.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <set>


struct MPIX_Data
{
    int num_msgs;
    int size_msgs; 
    std::vector<int> procs;
    std::vector<int> counts;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<MPI_Request> requests;
    std::vector<int> buffer;
};

template <typename T>
void standard_communication(std::vector<T>& send_vals, 
        std::vector<T>& recv_vals, int tag, MPI_Datatype mpi_type, 
        MPIX_Data* send_data, MPIX_Data* recv_data)
{
    int proc, start, end, idx;
    std::vector<T> buf(send_data->size_msgs);

    for (int i = 0; i < send_data->num_msgs; i++)
    {
        proc = send_data->procs[i];
        start = send_data->indptr[i];
        end = send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = send_data->indices[j];
            buf[j] = send_vals[idx];
        }
        MPI_Isend(&(buf[start]), end - start, mpi_type, proc, tag,
                MPI_COMM_WORLD, &send_data->requests[i]);
    }

    for (int i = 0; i < recv_data->num_msgs; i++)
    {
        proc = recv_data->procs[i];
        start = recv_data->indptr[i];
        end = recv_data->indptr[i+1];
        MPI_Irecv(&recv_vals[start], end - start, mpi_type, proc, tag, 
                MPI_COMM_WORLD, &recv_data->requests[i]);
    }
    
    if (send_data->num_msgs)
    {
        MPI_Waitall(send_data->num_msgs, send_data->requests.data(), 
                MPI_STATUSES_IGNORE);
    }
    if (recv_data->num_msgs)
    {
        MPI_Waitall(recv_data->num_msgs, recv_data->requests.data(), 
                MPI_STATUSES_IGNORE);
    }
} 

// Form random communication
void form_initial_communicator(int local_size, MPIX_Data* send_data, MPIX_Data* recv_data)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Declare Variables
    srand(49352034 + rank);
    int n_sends = (rand() % 10) + 1; // Between 1 and 10 msgs sent
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
        while (proc == rank)
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
        size = (rand() % local_size) + 1;
        for (int j = 0; j < size; j++)
        {
            send_data->indices.push_back(ctr++);
            if (ctr >= local_size) ctr = 0;
        }
        send_data->indptr[i+1] = send_data->indices.size();
        send_data->counts[i] = send_data->indptr[i+1] - send_data->indptr[i];
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
        start = send_data->indptr[i];
        end = send_data->indptr[i+1];
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
        recv_data->indptr[i+1] = recv_data->indptr[i] + size;
        recv_data->counts[i] = size;
    }
    recv_data->size_msgs = recv_data->indptr[recv_data->num_msgs];
    recv_data->buffer.resize(recv_data->size_msgs);

    if (send_data->num_msgs)
    {
        MPI_Waitall(send_data->num_msgs, send_data->requests.data(),
                MPI_STATUSES_IGNORE);
    }
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //


TEST(RandomCommTest, TestsInTests)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    setenv("PPN", "4", 1);

    // Initial communication info (standard)
    int local_size = 10000; // Number of variables each rank stores
    MPIX_Data send_data;
    MPIX_Data recv_data;
    form_initial_communicator(local_size, &send_data, &recv_data);

    // Determine unique global indices (map from send indices to recv indicies)
    int first = local_size * rank;
    std::vector<int> global_send_idx(send_data.size_msgs);
    std::vector<int> global_recv_idx(recv_data.size_msgs);
    int tag = 29354;
    for (int i = 0; i < send_data.num_msgs; i++)
    {
        int proc = send_data.procs[i];
        int start = send_data.indptr[i];
        int end = send_data.indptr[i+1];
        for (int j = start; j < end; j++)
        {
            int idx = send_data.indices[j];
            global_send_idx[j] = first + idx;
        }
        MPI_Isend(&global_send_idx[start], end - start, MPI_INT, proc,
                tag, MPI_COMM_WORLD, &send_data.requests[i]);
    }
    for (int i = 0; i < recv_data.num_msgs; i++)
    {
        int proc = recv_data.procs[i];
        int start = recv_data.indptr[i];
        int end = recv_data.indptr[i+1];
        MPI_Irecv(&global_recv_idx[start], end - start, MPI_INT, proc,
                tag, MPI_COMM_WORLD, &recv_data.requests[i]);
    }
    if (send_data.num_msgs) MPI_Waitall(send_data.num_msgs, 
            send_data.requests.data(), MPI_STATUSES_IGNORE);
    if (recv_data.num_msgs) MPI_Waitall(recv_data.num_msgs, 
            recv_data.requests.data(), MPI_STATUSES_IGNORE);

    // Initializing node-aware communication package
    MPIX_Comm* topo_comm;
    MPIX_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            recv_data.num_msgs, recv_data.procs.data(), recv_data.indptr.data(),
            global_recv_idx.data(),
            send_data.num_msgs, send_data.procs.data(),
            send_data.indptr.data(), send_data.indices.data(),
            global_send_idx.data(),
            MPI_INFO_NULL, 0, &topo_comm
    );

    // Test correctness of communication
    std::vector<int> send_vals(local_size);
    int val = local_size*rank;
    for (int i = 0; i < local_size; i++)
    {
        send_vals[i] = val++;
    }
    std::vector<int> std_recv_vals(recv_data.size_msgs);
    std::vector<int> nap_recv_vals(recv_data.size_msgs);

    // 1. Standard Communication
    standard_communication(send_vals, std_recv_vals, 49345, MPI_INT, &send_data, &recv_data);

    // 2. Node-Aware Communication
    MPIX_Neighbor_alltoallv(send_vals.data(), send_data.counts.data(),
            send_data.indptr.data(), MPI_INT,
            nap_recv_vals.data(), recv_data.counts.data(),
            recv_data.indptr.data(), MPI_INT,
            topo_comm);

    // 3. Compare std_recv_vals and nap_recv_vals
    for (int i = 0; i < recv_data.size_msgs; i++)
    {
        ASSERT_EQ(std_recv_vals[i], nap_recv_vals[i]);
    }
    
    std::vector<double> send_d_vals(local_size);
    val = local_size*rank;
    for (int i = 0; i < local_size; i++)
    {
        send_vals[i] = val++;
    }
    std::vector<double> std_recv_d_vals(recv_data.size_msgs);
    std::vector<double> nap_recv_d_vals(recv_data.size_msgs);

    // 1. Standard Communication
    standard_communication(send_d_vals, std_recv_d_vals, 49345, MPI_INT, &send_data, &recv_data);

    // 2. Node-Aware Communication
    MPIX_Neighbor_alltoallv(send_d_vals.data(), send_data.counts.data(),
            send_data.indptr.data(), MPI_INT,
            nap_recv_d_vals.data(), recv_data.counts.data(),
            recv_data.indptr.data(), MPI_INT,
            topo_comm);
    //MPIX_INAPsend(send_d_vals.data(), topo_comm->nap_comm, MPI_INT, 20423, MPI_COMM_WORLD);
    //MPIX_INAPrecv(nap_recv_d_vals.data(), topo_comm->nap_comm, MPI_INT, 20423, MPI_COMM_WORLD);
    //MPIX_NAPwait(topo_comm->nap_comm);

    // 3. Compare std_recv_vals and nap_recv_vals
    for (int i = 0; i < recv_data.size_msgs; i++)
    {
        ASSERT_NEAR(std_recv_d_vals[i], nap_recv_d_vals[i], 1e-10);
    }

    //MPIX_NAPDestroy(nap_comm);
    MPIX_Comm_free(topo_comm);
    free(topo_comm);

    setenv("PPN", "16", 1);
}

