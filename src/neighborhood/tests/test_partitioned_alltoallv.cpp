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
//#include <unistd.h>
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

void pack_partd(
    int n_vec,
    int n_sends,
    std::vector<int>& idxs,
    std::vector<int>& displs,
    std::vector<double>& x,
    std::vector<double>& send_buffer,
    std::vector<MPIP_Request>& sreqs
) {
    for (int i = 0; i < n_sends; i++) {
        #pragma omp for nowait schedule(static)
        for (int j = displs[i] * n_vec; j < displs[i+1] * n_vec; j++) {
            int idx = idxs[j / n_vec];
            send_buffer[j] = x[(idx * n_vec) + (j % n_vec)];
        }

        #pragma omp critical
        MPIP_Pready(omp_get_thread_num(), &sreqs[i]);
    }

    {
    // Serial packing code for reference
    // if (A.send_comm.size_msgs)
    // {
        // for (int i = 0; i < A.send_comm.size_msgs; i++)
        // {
        //     idx = A.send_comm.idx[i];
        //     for (int j = 0; j < n_vec; j++)
        //         partd_send_vals[(i * n_vec) + j] = x[(idx * n_vec) + j];
        // }
    // }
    }
}

void SpMV_threaded(
    Mat& A,
    std::vector<double>& x,
    std::vector<double>& b,
    int beta,
    int n_vec
) {
    int start, end;
    int data, col_idx;

    #pragma omp for private(start, end, data, col_idx)
    for (int i = 0; i < A.n_rows; i++)
    {
        start = A.rowptr[i];
        end = A.rowptr[i+1];
        for (int j = start; j < end; j  ++)
        {
            data = A.data[j];
            col_idx = A.col_idx[j];
            for (int vec = 0; vec < n_vec; vec++) {
                b[i * n_vec + vec] = (data * x[col_idx * n_vec + vec]) + \
                                        (beta * b[i * n_vec + vec]);
            }
        }
    }
}

void SpMV_off_proc_CSC( // single column
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
            #pragma omp atomic // TODO this makes the operation serial without considering which element of b is being accessed?
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
    #pragma omp parallel
    {
        SpMV_threaded(A.on_proc, x, b, 0, n_vec);
        #pragma omp master
        {
            communicate(A, x, recv_buff, MPI_DOUBLE, n_vec);
        }
        #pragma omp barrier
        SpMV_threaded(A.off_proc, recv_buff, b, 1, n_vec);
    }
}

void par_SpMV_partd(
    int n_vec,
    ParMat<int>& A,
    std::vector<double>& x,
    std::vector<double>& b,
    std::vector<double>& send_buff,
    std::vector<double>& x_off_proc,
    std::vector<MPIP_Request>& sreqs,
    std::vector<MPIP_Request>& rreqs
) {
    int n_sends = A.send_comm.n_msgs;
    int n_recvs = A.recv_comm.n_msgs;

    MPIP_Startall(n_sends, sreqs.data());
    MPIP_Startall(n_recvs, rreqs.data());

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #pragma omp parallel // TODO possibly move outside iterations
    {
        // Local compute
        SpMV_threaded(A.on_proc, x, b, 0, n_vec);

        // Pack and early send
        if (A.send_comm.size_msgs) {
            pack_partd(n_vec, n_sends, A.send_comm.idx, A.send_comm.ptr, x,
                        send_buff, sreqs);
        }

        // Receive
        //#pragma omp barrier

        #pragma omp master
        {
            if (n_recvs) {
                MPIP_Waitall(n_recvs, rreqs.data(), MPI_STATUSES_IGNORE);
            }
        }
        #pragma omp barrier

        // Non-local compute
    //#pragma omp parallel
    //{
        SpMV_threaded(A.off_proc, x_off_proc, b, 1, n_vec);
    //}
    } // implicit barrier

    MPIP_Waitall(n_sends, sreqs.data(), MPI_STATUSES_IGNORE);
}

void par_SpMV_partd_csc(
    int n_vec,
    ParMat<int>& A,
    Mat& A_csc,
    std::vector<double>& x,
    std::vector<double>& b,
    std::vector<double>& send_buff,
    std::vector<double>& x_off_proc,
    std::vector<MPIP_Request>& sreqs,
    std::vector<MPIP_Request>& rreqs
) {
    int rank; // TODO
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n_sends = A.send_comm.n_msgs;
    int n_recvs = A.recv_comm.n_msgs;

    MPIP_Startall(n_sends, sreqs.data());
    MPIP_Startall(n_recvs, rreqs.data());

    int elems_done = 0;
    int n_threads = omp_get_max_threads();
    #pragma omp parallel num_threads(n_threads) // TODO possibly move outside iterations
    {
        // Local compute
        SpMV_threaded(A.on_proc, x, b, 0, n_vec);

        // Pack and early send
        if (A.send_comm.size_msgs)
            pack_partd(n_vec, n_sends, A.send_comm.idx, A.send_comm.ptr, x,
                        send_buff, sreqs);

        // Receive and early compute
        int n_req = n_recvs;
        int next_n_req = 0;
        int flag;
        int part = omp_get_thread_num();
        int n_parts = omp_get_num_threads();
        int start, end, part_size, start_idx, start_row;
        int vec_row_start, vec_row_end, end_idx, end_row;
        int req_idxs[n_recvs];
        for (int i = 0; i < n_recvs; i++) {
            req_idxs[i] = i;
        }
        //printf("rank %d %d recving %d msgs\n", rank, part, A.recv_comm.n_msgs);

        while (n_req)
        {
            next_n_req = 0;
            for (int i = 0; i < n_req; i++)
            {
                //printf("%d\n", i);
                int req_idx = req_idxs[i];
                MPIP_Parrived(&rreqs[req_idx], part, &flag);
                if (flag) {
                    start = A.recv_comm.ptr[req_idx];
                    end   = A.recv_comm.ptr[req_idx+1];
                    part_size = (end - start) * n_vec / n_parts; // assumes n_parts divides n_vec
                    start_idx = (start * n_vec) + (part_size * part);
                    start_row = start_idx / n_vec;
                    vec_row_start = start_idx % n_vec; // first column received in first row
                    end_idx = (start * n_vec) + (part_size * (part + 1));
                    end_row = end_idx / n_vec;
                    vec_row_end = end_idx % n_vec;
                    if (vec_row_end == 0) { // TODO check this if/else
                        vec_row_end = n_vec;
                    } else {
                        end_row++;
                    }

                    // printf("rank %d %d msg %d, %d parts\n", rank, part, A.recv_comm.procs[req_idx], n_parts);
                    // printf("rank %d %d part_size %d, start_idx %d, start_row %d, end_idx %d, end_row %d\n",
                    //          rank, part, part_size, start_idx, start_row, end_idx, end_row);
                    // printf("rank %d %d start %d, end %d, msg n_vec_rows %d\n", rank, part, start, end, end-start);
                    // printf("rank %d %d ncols: %d/%d\n", rank, part, (end_row - start_row), A_csc.n_cols);

                    // possible race condition on b, see pragma above
                    for (int j = start_row; j < end_row; j++) {
                        int end_pos = n_vec;
                        if (j == end_row - 1) // only use vec_row_end for last vec row
                            end_pos = vec_row_end;
                        //printf("rank %d %d j: %d, vec_start %d, vec_end: %d\n", rank, part, j, vec_row_start, end_pos);
                        // #pragma omp critical
                        // {
                        SpMV_off_proc_CSC(n_vec, vec_row_start, end_pos, j, A_csc, x_off_proc, b);
                        // }
                        if (j == start_row) // only use vec_row_start for first vec row
                            vec_row_start = 0;
                    }
                    #pragma omp atomic
                    elems_done += part_size;
                } else {
                    req_idxs[next_n_req++] = req_idx;
                }
            }
            n_req = next_n_req;
        }
    }

    // TODO remove. Checking no missed elements
    if (elems_done != A_csc.n_cols * n_vec) {
        printf("rank %d missed elems: %d. %d/%d\n", rank, A_csc.n_cols * n_vec - elems_done, elems_done, A_csc.n_cols * n_vec);
        assert(1==0);
    }

    // // TODO remove
    // for (int i = 0; i < A_csc.n_cols; i++) {
    //     SpMV_off_proc_CSC(n_vec, 0, n_vec, i, A_csc, x_off_proc, b);
    // }

    MPIP_Waitall(n_sends, sreqs.data(), MPI_STATUSES_IGNORE);
    MPIP_Waitall(n_recvs, rreqs.data(), MPI_STATUSES_IGNORE); // TODO try deleting
}

// Test compares 3 methods:
// 1. Baseline non-partitioned approach (std_recv_vals, x1, b1...)
// 2. Partitioned approach              (partd_send_vals, partd_recv_vals, x2, b2...)
// 3. Partitioned approach using CSC    (partd_csc_send_vals, partd_csc_recv_vals, x3, b3...)
void test_partitioned(const char* filename, int n_vec)
{
    // msg counts must be divisible by n_parts. simplest method is to have n_parts divide n_vec
    // TODO assert this ^
    int n_parts = omp_get_max_threads(); // TODO check this if things break

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

    std::vector<double> std_recv_vals, partd_recv_vals, partd_csc_recv_vals;
    std::vector<double> send_vals, partd_send_vals, partd_csc_send_vals;
    std::vector<double> x1, x2, x3, b1, b2, b3;

    if (A.on_proc.n_cols)
    {
        send_vals.resize(A.on_proc.n_cols * n_vec);
        std::iota(send_vals.begin(), send_vals.end(), 0);

        x1.resize(A.on_proc.n_rows * n_vec);
        x2.resize(A.on_proc.n_cols * n_vec);
        x3.resize(A.on_proc.n_cols * n_vec);
        b1.resize(A.on_proc.n_rows * n_vec);
        b2.resize(A.on_proc.n_rows * n_vec);
        b3.resize(A.on_proc.n_rows * n_vec);
        for (int i = 0; i < A.on_proc.n_cols * n_vec; i++)
        {
            send_vals[i] += (rank*1000);
            x1[i] = send_vals[i];
            x2[i] = send_vals[i];
            x3[i] = send_vals[i];
        }
    }

    if (A.recv_comm.size_msgs)
    {
        std_recv_vals.resize(A.recv_comm.size_msgs * n_vec);
        partd_recv_vals.resize(A.recv_comm.size_msgs * n_vec);
        partd_csc_recv_vals.resize(A.recv_comm.size_msgs * n_vec);
    }

    if (A.send_comm.size_msgs)
    {
        partd_send_vals.resize(A.send_comm.size_msgs * n_vec);
        partd_csc_send_vals.resize(A.send_comm.size_msgs * n_vec);
    }

    Mat A_off_proc_CSC;
    CSR_to_CSC(A.off_proc, A_off_proc_CSC);


    // Initialization
    std::vector<MPIP_Request> sreqs;
    std::vector<MPIP_Request> rreqs;
    std::vector<MPIP_Request> sreqs_csc;
    std::vector<MPIP_Request> rreqs_csc;
    if (A.send_comm.n_msgs) {
        sreqs.resize(A.send_comm.n_msgs);
        sreqs_csc.resize(A.send_comm.n_msgs);
    }
    if (A.recv_comm.n_msgs) {
        rreqs.resize(A.recv_comm.n_msgs);
        rreqs_csc.resize(A.recv_comm.n_msgs);
    }

    int proc;
    int start, end;
    int tag = 2949;
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc = A.recv_comm.procs[i];
        start = A.recv_comm.ptr[i] * n_vec;
        end = A.recv_comm.ptr[i+1] * n_vec;
        MPIP_Precv_init(&(partd_recv_vals[start]), n_parts, (int)(end - start)/n_parts, MPI_DOUBLE, proc, tag,
                MPI_COMM_WORLD, MPI_INFO_NULL, &(rreqs[i]));
        MPIP_Precv_init(&(partd_csc_recv_vals[start]), n_parts, (int)(end - start)/n_parts, MPI_DOUBLE, proc, tag,
                MPI_COMM_WORLD, MPI_INFO_NULL, &(rreqs_csc[i]));
    }

    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        proc = A.send_comm.procs[i];
        start = A.send_comm.ptr[i] * n_vec;
        end = A.send_comm.ptr[i+1] * n_vec;
        MPIP_Psend_init(&(partd_send_vals[start]), n_parts, (int)(end - start)/n_parts, MPI_DOUBLE, proc, tag,
                MPI_COMM_WORLD, MPI_INFO_NULL, &(sreqs[i]));
        MPIP_Psend_init(&(partd_csc_send_vals[start]), n_parts, (int)(end - start)/n_parts, MPI_DOUBLE, proc, tag,
                MPI_COMM_WORLD, MPI_INFO_NULL, &(sreqs_csc[i]));
    }


    // Test Iterations
    int test_iters = 10; // TODO add updating and resetting of x after tests
    for (int iter = 0; iter < test_iters; iter++) {
        par_SpMV(n_vec, A, x1, b1, std_recv_vals);

        par_SpMV_partd(n_vec, A, x2, b2, partd_send_vals, partd_recv_vals, sreqs, rreqs);

        par_SpMV_partd_csc(n_vec, A, A_off_proc_CSC, x3, b3, partd_csc_send_vals,
                            partd_csc_recv_vals, sreqs_csc, rreqs_csc);

        for (int i = 0; i < A.recv_comm.size_msgs; i++)
        {
            if (fabs(std_recv_vals[i] - partd_recv_vals[i]) / std_recv_vals[i] > 1e-9) {
                printf("rank %d DIFF in recv vals at pos %d, std %e, partd %e, diff: %e\n",
                        rank, i, std_recv_vals[i], partd_recv_vals[i],
                        fabs(std_recv_vals[i] - partd_recv_vals[i]));
                assert(1 == 0);
            }
        }
        for (size_t i = 0; i < b1.size(); i++)
        {
            if (fabs(b1[i] - b2[i]) / b1[i] > 1e-9) {
                printf("rank %d DIFF in b at pos %d, std %e, partd %e, diff: %e\n", rank, i, b1[i], b2[i], fabs(b1[i] - b2[i]));
                assert(1 == 0);
            }
        }
        for (int i = 0; i < A.recv_comm.size_msgs; i++)
        {
            if (fabs(std_recv_vals[i] - partd_csc_recv_vals[i]) / std_recv_vals[i] > 1e-9) {
                printf("rank %d DIFF in recv vals at pos %d, std %e, csc %e, diff: %e\n",
                        rank, i, std_recv_vals[i], partd_csc_recv_vals[i],
                        fabs(std_recv_vals[i] - partd_csc_recv_vals[i]));
                assert(1 == 0);
            }
        }
        for (size_t i = 0; i < b1.size(); i++)
        {
            if (fabs(b1[i] - b3[i]) / b1[i] > 1e-9) {
                printf("rank %d DIFF in b at pos %d, std %e, csc %e, diff: %e\n", rank, i, b1[i], b3[i], fabs(b1[i] - b3[i]));
                printf("b2: %e\n", b2[i]);
                assert(1 == 0);
            }
        }

        std::swap(x1, b1);
        std::swap(x2, b2);
        std::swap(x3, b3);
    }


    // Estimate test iterations needed
    double t0, tf;
    int iters1, iters2, iters3;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < test_iters; i++) {
        par_SpMV(n_vec, A, x1, b1, std_recv_vals);
        std::swap(x1, b1);
    }
    tf = MPI_Wtime() - t0;
    MPI_Allreduce(&tf, &t0, 1, MPI_DOUBLE, MPI_MAX,
        MPI_COMM_WORLD);
    iters1 = (1.0 / (t0/test_iters)) + 1;
    if ((t0/test_iters) > 1.0)
        iters1 = 1;

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < test_iters; i++) {
        par_SpMV_partd(n_vec, A, x2, b2, partd_send_vals, partd_recv_vals, sreqs, rreqs);
        std::swap(x2, b2);
    }
    tf = MPI_Wtime() - t0;
    MPI_Allreduce(&tf, &t0, 1, MPI_DOUBLE, MPI_MAX,
        MPI_COMM_WORLD);
    iters2 = (1.0 / (t0/test_iters)) + 1;
    if ((t0/test_iters) > 1.0)
        iters2 = 1;

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < test_iters; i++) {
        par_SpMV_partd_csc(n_vec, A, A_off_proc_CSC, x3, b3, partd_csc_send_vals,
                            partd_csc_recv_vals, sreqs_csc, rreqs_csc);
        std::swap(x3, b3);
    }
    tf = MPI_Wtime() - t0;
    MPI_Allreduce(&tf, &t0, 1, MPI_DOUBLE, MPI_MAX,
        MPI_COMM_WORLD);
    iters3 = (1.0 / (t0/test_iters)) + 1;
    if ((t0/test_iters) > 1.0)
        iters3 = 1;

    if (rank == 0)
        printf("iters1 %d, iters2 %d, iters3 %d\n", iters1, iters2, iters3);


    // Reset x's from tests
    for (int i = 0; i < A.on_proc.n_cols * n_vec; i++)
    {
        x1[i] = send_vals[i];
        x2[i] = send_vals[i];
        x3[i] = send_vals[i];
    }

    //return;

    // Timing Iterations
    // Baseline
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int iter = 0; iter < iters1; iter++) {
        par_SpMV(n_vec, A, x1, b1, std_recv_vals);
        std::swap(x1, b1); // update x
    }
    tf = MPI_Wtime() - t0;
    MPI_Reduce(&tf, &t0, 1, MPI_DOUBLE, MPI_MAX, 0,
        MPI_COMM_WORLD);
    if (rank == 0) printf("Baseline Time for %d vectors: %e\n", n_vec, t0/iters1);


    // Partitioned
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int iter = 0; iter < iters2; iter++) {
        par_SpMV_partd(n_vec, A, x2, b2, partd_send_vals, partd_recv_vals, sreqs, rreqs);
        std::swap(x2, b2); // update x
    }
    tf = MPI_Wtime() - t0;
    MPI_Reduce(&tf, &t0, 1, MPI_DOUBLE, MPI_MAX, 0,
        MPI_COMM_WORLD);
    if (rank == 0) printf("Partitioned Time for %d vectors: %e\n", n_vec, t0/iters2);


    // Partitioned CSC
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int iter = 0; iter < iters3; iter++) {
        par_SpMV_partd_csc(n_vec, A, A_off_proc_CSC, x3, b3, partd_csc_send_vals,
                            partd_csc_recv_vals, sreqs_csc, rreqs_csc);
        std::swap(x3, b3); // update x
    }
    tf = MPI_Wtime() - t0;
    MPI_Reduce(&tf, &t0, 1, MPI_DOUBLE, MPI_MAX, 0,
        MPI_COMM_WORLD);
    if (rank == 0) printf("Partitioned CSC Time for %d vectors: %e\n", n_vec, t0/iters3);


    // Cleanup
    for (int i = 0; i < A.recv_comm.n_msgs; i++) {
        MPIP_Request_free(&rreqs[i]);
        MPIP_Request_free(&rreqs_csc[i]);
    }
    for (int i = 0; i < A.send_comm.n_msgs; i++) {
        MPIP_Request_free(&sreqs[i]);
        MPIP_Request_free(&sreqs_csc[i]);
    }
}

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        printf("no mpi thread multiple\n");
        assert(1==0);
    }
    
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
            test_partitioned((mat_dir + test_matrices[i]).c_str(), vec_sizes[j]);
        }
    }

    MPI_Finalize();
}
