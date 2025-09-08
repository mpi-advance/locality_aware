#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <numeric>
#include <set>

#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"

void compare_neighbor_alltoallv_results(std::vector<int>& pmpi_recv_vals, std::vector<int>& mpix_recv_vals, int s)
{
    for (int i = 0; i < s; i++)
    {
        if (pmpi_recv_vals[i] != mpix_recv_vals[i])
        {
            fprintf(stderr, "PMPI recv != MPIX: position %d, pmpi %d, mpix %d\n", i, 
                    pmpi_recv_vals[i], mpix_recv_vals[i]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void test_matrix(const char* filename)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Read suitesparse matrix
    ParMat<int> A;
    int idx, ctr, proc, size;
    readParMatrix(filename, A);
    form_comm(A);

    // Create list of all processes
    std::vector<int> procs(num_procs);
    std::iota(procs.begin(), procs.end(), 0);
    std::vector<int> proc_send_pos(num_procs);
    std::vector<int> proc_recv_pos(num_procs);

    std::vector<int> proc_send_sizes(num_procs, 0);
    std::vector<int> proc_recv_sizes(num_procs, 0);
    std::vector<int> proc_send_displs(num_procs+1);
    std::vector<int> proc_recv_displs(num_procs+1);
    std::vector<long> proc_send_indices(A.send_comm.size_msgs);
    std::vector<long> proc_recv_indices(A.recv_comm.size_msgs);

    std::vector<int> pmpi_recv_vals, mpix_recv_vals;
    std::vector<int> send_vals, alltoallv_send_vals;

    if (A.on_proc.n_cols)
    {
        send_vals.resize(A.on_proc.n_cols);
        std::iota(send_vals.begin(), send_vals.end(), 0);
        for (int i = 0; i < A.on_proc.n_cols; i++)
            send_vals[i] += (rank*1000);
    }

    if (A.recv_comm.size_msgs)
    {
        pmpi_recv_vals.resize(A.recv_comm.size_msgs);
        mpix_recv_vals.resize(A.recv_comm.size_msgs);
    }

    if (A.send_comm.size_msgs)
    {
        alltoallv_send_vals.resize(A.send_comm.size_msgs);
    }


    // Create dense communication graph
    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        proc = A.send_comm.procs[i];
        size = A.send_comm.counts[i];
        proc_send_sizes[proc] = size;
        proc_send_pos[proc] = i;
    }
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc = A.recv_comm.procs[i];
        size = A.recv_comm.counts[i];
        proc_recv_sizes[proc] = size;
        proc_recv_pos[proc] = i;
    }
    proc_send_displs[0] = 0;
    proc_recv_displs[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        ctr = proc_send_displs[i];
        if (proc_send_sizes[i])
        {
            idx = proc_send_pos[i];
            for (int j = A.send_comm.ptr[idx]; j < A.send_comm.ptr[idx+1]; j++)
            {
                alltoallv_send_vals[ctr] = send_vals[A.send_comm.idx[j]];
                proc_send_indices[ctr++] = A.send_comm.idx[j] + A.first_col;
            }
        }
        proc_send_displs[i+1] = ctr;

        ctr = proc_recv_displs[i];
        if (proc_recv_sizes[i])
        {
            idx = proc_recv_pos[i];
            for (int j = A.recv_comm.ptr[idx]; j < A.recv_comm.ptr[idx+1]; j++)
            {
                proc_recv_indices[ctr++] = A.off_proc_columns[j];
            }
        }
        proc_recv_displs[i+1] = ctr;
    }

    // MPI and MPIX Variables
    MPI_Status status;
    MPIX_Comm* xcomm;
    MPIX_Request* xrequest;
    MPIX_Info* xinfo;
    MPIX_Info_init(&xinfo);


    // Create standard PMPI neighbor communicator
    MPI_Comm std_comm;
    PMPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            num_procs,
            procs.data(),
            MPI_UNWEIGHTED,
            num_procs, 
            procs.data(),
            MPI_UNWEIGHTED,
            MPI_INFO_NULL, 
            0, 
            &std_comm);

    // Standard PMPI neighbor exchange
    PMPI_Neighbor_alltoallv(alltoallv_send_vals.data(),
            proc_send_sizes.data(),
            proc_send_displs.data(), 
            MPI_INT,
            pmpi_recv_vals.data(),
            proc_recv_sizes.data(),
            proc_recv_displs.data(),
            MPI_INT,
            std_comm);

    PMPI_Comm_free(&std_comm);


    // MPI Advance neighbor communicator
    MPIX_Dist_graph_create_adjacent(MPI_COMM_WORLD,
            num_procs,
            procs.data(),
            MPI_UNWEIGHTED,
            num_procs, 
            procs.data(),
            MPI_UNWEIGHTED,
            xinfo, 
            0, 
            &xcomm);
    update_locality(xcomm, 4);
    

    // Standard exchange
    mpix_neighbor_alltoallv_implementation = NEIGHBOR_ALLTOALLV_STANDARD;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIX_Neighbor_alltoallv(alltoallv_send_vals.data(), 
            proc_send_sizes.data(),
            proc_send_displs.data(), 
            MPI_INT,
            mpix_recv_vals.data(), 
            proc_recv_sizes.data(),
            proc_recv_displs.data(), 
            MPI_INT,
            xcomm);
    compare_neighbor_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);


    // 2. Node-Aware Communication
    mpix_neighbor_alltoallv_init_implementation = NEIGHBOR_ALLTOALLV_INIT_STANDARD;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIX_Neighbor_alltoallv_init(alltoallv_send_vals.data(), 
            proc_send_sizes.data(),
            proc_send_displs.data(), 
            MPI_INT,
            mpix_recv_vals.data(), 
            proc_recv_sizes.data(),
            proc_recv_displs.data(), 
            MPI_INT,
            xcomm, 
            xinfo,
            &xrequest);
            

    MPIX_Start(xrequest);
    MPIX_Wait(xrequest, &status);
    MPIX_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);


    // 3. MPI Advance - Optimized Communication
    mpix_neighbor_alltoallv_init_implementation = NEIGHBOR_ALLTOALLV_INIT_LOCALITY;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIX_Neighbor_alltoallv_init(alltoallv_send_vals.data(), 
            proc_send_sizes.data(),
            proc_send_displs.data(), 
            MPI_INT,
            mpix_recv_vals.data(), 
            proc_recv_sizes.data(),
            proc_recv_displs.data(), 
            MPI_INT,
            xcomm, 
            xinfo,
            &xrequest);

    MPIX_Start(xrequest);
    MPIX_Wait(xrequest, &status);
    MPIX_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);

    // Standard from Extended Interface
    mpix_neighbor_alltoallv_init_implementation = NEIGHBOR_ALLTOALLV_INIT_STANDARD;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIX_Neighbor_alltoallv_init_ext(alltoallv_send_vals.data(), 
            proc_send_sizes.data(),
            proc_send_displs.data(), 
            proc_send_indices.data(),
            MPI_INT,
            mpix_recv_vals.data(), 
            proc_recv_sizes.data(),
            proc_recv_displs.data(), 
            proc_recv_indices.data(),
            MPI_INT,
            xcomm, 
            xinfo,
            &xrequest);
    MPIX_Start(xrequest);
    MPIX_Wait(xrequest, &status);
    MPIX_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);

    // Full Locality
    mpix_neighbor_alltoallv_init_implementation = NEIGHBOR_ALLTOALLV_INIT_LOCALITY;
    std::fill(mpix_recv_vals.begin(), mpix_recv_vals.end(), 0);
    MPIX_Neighbor_alltoallv_init_ext(alltoallv_send_vals.data(), 
            proc_send_sizes.data(),
            proc_send_displs.data(), 
            proc_send_indices.data(),
            MPI_INT,
            mpix_recv_vals.data(), 
            proc_recv_sizes.data(),
            proc_recv_displs.data(), 
            proc_recv_indices.data(),
            MPI_INT,
            xcomm, 
            xinfo,
            &xrequest);
    MPIX_Start(xrequest);
    MPIX_Wait(xrequest, &status);
    MPIX_Request_free(&xrequest);
    compare_neighbor_alltoallv_results(pmpi_recv_vals, mpix_recv_vals, A.recv_comm.size_msgs);

    MPIX_Info_free(&xinfo);
    MPIX_Comm_free(&xcomm);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    test_matrix("../../../../test_data/dwt_162.pm");
    test_matrix("../../../../test_data/odepa400.pm");
    test_matrix("../../../../test_data/ww_36_pmec_36.pm");
    test_matrix("../../../../test_data/bcsstk01.pm");
    test_matrix("../../../../test_data/west0132.pm");
    test_matrix("../../../../test_data/gams10a.pm");
    test_matrix("../../../../test_data/gams10am.pm");
    test_matrix("../../../../test_data/D_10.pm");
    test_matrix("../../../../test_data/oscil_dcop_11.pm");
    test_matrix("../../../../test_data/tumorAntiAngiogenesis_4.pm");
    test_matrix("../../../../test_data/ch5-5-b1.pm");
    test_matrix("../../../../test_data/msc01050.pm");
    test_matrix("../../../../test_data/SmaGri.pm");
    test_matrix("../../../../test_data/radfr1.pm");
    test_matrix("../../../../test_data/bibd_49_3.pm");
    test_matrix("../../../../test_data/can_1054.pm");
    test_matrix("../../../../test_data/can_1072.pm");
    test_matrix("../../../../test_data/lp_sctap2.pm");
    test_matrix("../../../../test_data/lp_woodw.pm");
    
    MPI_Finalize();
    return 0;
} // end of main() //



