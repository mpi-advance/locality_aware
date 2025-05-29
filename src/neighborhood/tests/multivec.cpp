// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails
// while EXPECT_* variants continue with the run.

// OUTLINE:
// Precv/Psend inits
// Launch threads?
// set x to be send_vals
// Iterations: use x as send_vals
    // Communicate: For each message:
        // Start
        // Pack
        // Pready's
        // Wait/parrives?
        // unpack
    // Compute b
    // set x to b

#include "mpi_advance.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <numeric>
#include <set>
#include <string>
#include <omp.h>

#include "tests/sparse_mat.hpp"
#include "tests/par_binary_IO.hpp"
#include "mpipcl.h"


void CSR_to_CSC(Mat& A, Mat& B)
{
    B.n_rows = A.n_rows;
    B.n_cols = A.n_cols;
    B.nnz = A.nnz;

    // Resize vectors to appropriate dimensions
    B.rowptr.resize(B.n_cols+1, 0);
    B.col_idx.resize(B.nnz);
    B.data.resize(B.nnz);

    // Create indptr, summing number times row appears in CSC
    for (int i = 0; i < A.nnz; i++)
    {
        B.rowptr[A.col_idx[i] + 1]++;
    }
    for (int i = 1; i <= A.n_cols; i++)
    {
        B.rowptr[i] += B.rowptr[i-1];
    }

    // Add values to indices and data
    std::vector<int> ctr(B.n_cols, 0);
    for (int i = 0; i < A.n_rows; i++)
    {
        int row_start = A.rowptr[i];
        int row_end = A.rowptr[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            int col = A.col_idx[j];
            int idx = B.rowptr[col] + ctr[col]++;
            B.col_idx[idx] = i;
            B.data[idx] = A.data[j];
        }
    }
}

void SpMV_threaded(
    int n_vec,
    Mat& A,
    std::vector<double>& x,
    std::vector<double>& b
) {
    int start, end;
    int data, col;

    #pragma omp for private(start, end, data, col)
    for (int row = 0; row < A.n_rows; row++)
    {
        start = A.rowptr[row];
        end = A.rowptr[row+1];
        for (int j = start; j < end; j  ++)
        {
            data = A.data[j];
            col = A.col_idx[j];
            for (int vec = 0; vec < n_vec; vec++) {
                b[row * n_vec + vec] += data * x[col * n_vec + vec];
            }
        }
    }
}

void SpMV_off_proc_CSC( // single column
    int n_vec,
    Mat& A,
    std::vector<double>& x,
    std::vector<double>& b
) {
    int start, end;
    int data, row;

    #pragma omp for private(start, end, data, row)
    for (int col = 0; col < A.n_cols; col++)
    {
        start = A.rowptr[col];
        end = A.rowptr[col+1];
        for (int j = start; j < end; j  ++)
        {
            data = A.data[j];
            row = A.col_idx[j];
            for (int vec = 0; vec < n_vec; vec++) {
                #pragma omp atomic
                b[row * n_vec + vec] += data * x[col * n_vec + vec];
            }
        }
    }
}

void SpMV_off_proc_CSC_part( // single column
    int n_vec,
    int vec_row_start,
    int vec_row_end,
    int col,
    Mat& A,
    std::vector<double>& x,
    std::vector<double>& b
) {
    int start, end;
    int data, row_idx;

    start = A.rowptr[col]; // rowptr is Column ptr for CSC
    end = A.rowptr[col+1];
    for (int i = start; i < end; i++)
    {
        data = A.data[i];
        row_idx = A.col_idx[i]; // col_idx is Row idx for CSC
        for (int vec = vec_row_start; vec < vec_row_end; vec++) {
            #pragma omp atomic
            b[row_idx * n_vec + vec] += data * x[col * n_vec + vec];
        }
    }
}



void par_SpMV(
    int n_vec,
    ParMat<int>& A,
    std::vector<double>& x,
    std::vector<double>& b,
    std::vector<double>& recv_buff
) {
    std::fill(b.begin(), b.end(), 0);
    #pragma omp parallel
    {
        SpMV_threaded(n_vec, A.on_proc, x, b);
        #pragma omp single
        {
            communicate(A, x, recv_buff, MPI_DOUBLE, n_vec);
        } // implicit barrier
        SpMV_threaded(n_vec, A.off_proc, recv_buff, b);
    }
}

void par_SpMV_CSC(
    int n_vec,
    ParMat<int>& A,
    Mat& A_csc,
    std::vector<double>& x,
    std::vector<double>& b,
    std::vector<double>& recv_buff
) {
    std::fill(b.begin(), b.end(), 0);
    #pragma omp parallel
    {

        SpMV_threaded(n_vec, A.on_proc, x, b);
        #pragma omp single
        {
            communicate(A, x, recv_buff, MPI_DOUBLE, n_vec);
        } // implicit barrier
        SpMV_off_proc_CSC(n_vec, A_csc, recv_buff, b);
    }
}

void par_SpMV_CSC_part(
    int n_vec,
    ParMat<int>& A,
    Mat& A_csc,
    std::vector<double>& x,
    std::vector<double>& b,
    std::vector<double>& recv_buff
) {
    std::fill(b.begin(), b.end(), 0);
    #pragma omp parallel
    {

        SpMV_threaded(n_vec, A.on_proc, x, b);
        #pragma omp single
        {
            communicate(A, x, recv_buff, MPI_DOUBLE, n_vec);
        } // implicit barrier
        #pragma omp for
        for (int col = 0; col < A_csc.n_cols; col++)
            SpMV_off_proc_CSC_part(n_vec, 0, n_vec, col, A_csc, recv_buff, b);
    }
}

void test_CSC(const char* filename, int n_vec)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Read suitesparse matrix
    ParMat<int> A;
    readParMatrix(filename, A);
    form_comm(A);
    if (rank == 0 && A.global_rows != A.global_cols) {
        printf("NOT SQUARE~~~~~~~~~~~~~~~~~~~~\n");
        return;
    }

    std::vector<double> std_recv_vals, new_recv_vals;
    std::vector<double> send_vals;
    std::vector<double> x, b_std, b_new;

    if (A.on_proc.n_cols)
    {
        send_vals.resize(A.on_proc.n_cols * n_vec);
        std::iota(send_vals.begin(), send_vals.end(), 0);

        x.resize(A.on_proc.n_rows * n_vec);
        b_std.resize(A.on_proc.n_rows * n_vec);
        b_new.resize(A.on_proc.n_rows * n_vec);

        for (int i = 0; i < A.on_proc.n_cols * n_vec; i++)
        {
            send_vals[i] += (rank*1000);
            x[i] = send_vals[i];
        }
    }

    if (A.recv_comm.size_msgs)
    {
        std_recv_vals.resize(A.recv_comm.size_msgs * n_vec);
        new_recv_vals.resize(A.recv_comm.size_msgs * n_vec);
    }

    // Convert off_proc to CSC
    Mat A_off_proc_CSC;
    CSR_to_CSC(A.off_proc, A_off_proc_CSC);

    // Compare CSR SpMV to CSC Spmv (all serial)
    std::iota(std_recv_vals.begin(), std_recv_vals.end(), 0);
    std::iota(new_recv_vals.begin(), new_recv_vals.end(), 0);
    std::fill(b_std.begin(), b_std.end(), 0.0);
    std::fill(b_new.begin(), b_new.end(), 0.0);
    SpMV_threaded(n_vec, A.off_proc, std_recv_vals, b_std);
    SpMV_off_proc_CSC(n_vec, A_off_proc_CSC, new_recv_vals, b_new);
    for (int i = 0; i < b_std.size(); i++)
        if (fabs(b_std[i] - b_new[i]) > 1e-06)
            printf("SERIAL CSC: DIFF at pos %d, std %e, new %e\n", i, b_std[i], b_new[i]);  


    // Compare ParCSR SpMV to ParCSC SpMV
    std::fill(b_std.begin(), b_std.end(), 0.0);
    std::fill(b_new.begin(), b_new.end(), 0.0);
    par_SpMV(n_vec, A, x, b_std, std_recv_vals);
    par_SpMV_CSC(n_vec, A, A_off_proc_CSC, x, b_new, new_recv_vals);
    for (int i = 0; i < A.recv_comm.size_msgs; i++)
        if (fabs(std_recv_vals[i] - new_recv_vals[i]) > 1e-06)
            printf("Diff in recv vals at pos %d!\n", i);
    for (int i = 0; i < b_std.size(); i++)
        if (fabs(b_std[i] - b_new[i]) > 1e-06)
            printf("PAR CSC: DIFF at pos %d, std %e, new %e\n", i, b_std[i], b_new[i]); 

}

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::string mat_dir = "../../../../test_data/";
    std::vector<std::string> test_matrices = {
        "dwt_162.pm",
        "odepa400.pm",
        "ww_36_pmec_36.pm",
        "bcsstk01.pm",
        "west0132.pm",
        "oscil_dcop_11.pm",
        "tumorAntiAngiogenesis_4.pm",
        "msc01050.pm",
        "SmaGri.pm",
        "radfr1.pm",
        "can_1054.pm",
        "can_1072.pm",
    };

    // Test SpM-Multivector
    std::vector<int> vec_sizes = {64};
    for (size_t i = 0; i < test_matrices.size(); i++) {
        if (rank == 0) 
            printf("Matrix %d...\n", i);
        for (size_t j = 0; j < vec_sizes.size(); j++) {
            test_CSC((mat_dir + test_matrices[i]).c_str(), vec_sizes[j]);
        }
    }

    MPI_Finalize();
}


