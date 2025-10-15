#include <mpi.h>
#include <vector>
#include <numeric>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "/g/g92/enamug/clean/GPU_locality_aware/locality_aware/src/tests/sparse_mat.hpp"
#include "/g/g92/enamug/clean/GPU_locality_aware/locality_aware/src/tests/par_binary_IO.hpp"
#include "/g/g92/enamug/clean/GPU_locality_aware/locality_aware/src/mpi_advance.h"


 void test_matrix(const char* filename, int n_iter)
{
    int rank, P; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &P);

    // Load matrix & form comm pattern
    ParMat<int> A;
    readParMatrix(filename, A);
    form_comm(A);

    
    std::vector<int> send_vals(A.on_proc.n_rows);
    for (int i = 0; i < A.on_proc.n_rows; ++i) send_vals[i] = rank*1000 + i;

    
     // Alltoallv_send_vals must be ordered (dest 0 to num_procs-1)
    std::vector<int> proc_pos(P, -1);
    for (int i = 0; i < A.send_comm.n_msgs; ++i)
        proc_pos[A.send_comm.procs[i]] = i;

    
    std::vector<int> packed_send(A.send_comm.size_msgs);//packing actual data to send
    int ctr = 0;
    for (int dest = 0; dest < P; ++dest) {
        int idx = proc_pos[dest];
        if (idx < 0) continue;
        int start = A.send_comm.ptr[idx];
        int end   = A.send_comm.ptr[idx+1];
        for (int j = start; j < end; ++j) {
            int row = A.send_comm.idx[j];
            assert(row >= 0 && row < (int)send_vals.size());
            packed_send[ctr++] = send_vals[row];
        }
    }
   

    // counts & displacements
    std::vector<int> sendcounts(P, 0), recvcounts(P, 0);
    std::vector<int> sdispls(P+1),     rdispls(P+1);

    for (int i = 0; i < A.send_comm.n_msgs; ++i)
        sendcounts[A.send_comm.procs[i]] = A.send_comm.ptr[i+1] - A.send_comm.ptr[i];
    for (int i = 0; i < A.recv_comm.n_msgs; ++i)
        recvcounts[A.recv_comm.procs[i]] = A.recv_comm.ptr[i+1] - A.recv_comm.ptr[i];

    sdispls[0] = 0; rdispls[0] = 0;
    for (int p = 0; p < P; ++p) {
        sdispls[p+1] = sdispls[p] + sendcounts[p];
        rdispls[p+1] = rdispls[p] + recvcounts[p];
    }
   

    /***This is the end of the original top sparse pattern ****/

    if (rdispls.back() == 0 && sdispls.back() == 0) {
        if (rank == 0) std::printf("No messages to exchange (empty pattern). Done.\n");
        return;
    }

    // ----- PMPI baseline alltoallv--------
    std::vector<int> recv_vals(rdispls.back(), 0);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int k = 0; k < n_iter; ++k) {
        PMPI_Alltoallv(packed_send.data(),
                       sendcounts.data(), sdispls.data(), MPI_INT,
                       recv_vals.data(),
                       recvcounts.data(), rdispls.data(), MPI_INT,
                       MPI_COMM_WORLD);
    }
    double pmpi_avg = (MPI_Wtime() - t0) / n_iter, pmpi_max = 0.0;
    MPI_Reduce(&pmpi_avg, &pmpi_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // bytes  rank 0 sends in total
    size_t bytes = (size_t)sdispls.back() * sizeof(int);
    if (rank == 0) {
        std::printf("[PMPI] Alltoallv avg: %e s, bytes/rank: %zu\n", pmpi_max, bytes);
    }

    // ---RMA winfence persistent --------
    MPIX_Comm*    xcomm = nullptr;
    MPIX_Info*    xinfo = nullptr;
    MPIX_Request* req   = nullptr;

    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
    update_locality(xcomm, 4);  
    MPIX_Info_init(&xinfo);

    std::vector<int> rma_recv(rdispls.back(), 0);

    // measure init+free average 
    t0 = MPI_Wtime();
    for (int k = 0; k < n_iter; ++k) {
        alltoallv_rma_winfence_init(packed_send.data(),
                                    sendcounts.data(), sdispls.data(), MPI_INT,
                                    rma_recv.data(),
                                    recvcounts.data(), rdispls.data(), MPI_INT,
                                    xcomm, xinfo, &req);
        MPIX_Request_free(req);
    }
    double rma_initavg = (MPI_Wtime() - t0) / n_iter, rma_initmax = 0.0;
    MPI_Reduce(&rma_initavg, &rma_initmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("[RMA ] winfence init+free in loop avg: %e s\n", rma_initmax);
    }

    // persistent request once
    alltoallv_rma_winfence_init(packed_send.data(),
                                sendcounts.data(), sdispls.data(), MPI_INT,
                                rma_recv.data(),
                                recvcounts.data(), rdispls.data(), MPI_INT,
                                xcomm, xinfo, &req);

    MPI_Barrier(xcomm->global_comm);
    t0 = MPI_Wtime();
    for (int k = 0; k < n_iter; ++k) {
        MPIX_Start(req);
        MPIX_Wait(req, MPI_STATUS_IGNORE);
    }
    double rma_avg = (MPI_Wtime() - t0) / n_iter, rma_max = 0.0;
    MPI_Reduce(&rma_avg, &rma_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("[RMA ] winfence persistent avg: %e s\n", rma_max);
    }

    // correctness vs PMPI
    if (rma_recv.size() != recv_vals.size()) {
        if (rank == 0) std::fprintf(stderr, "Size mismatch between PMPI and RMA recv buffers\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (size_t i = 0; i < rma_recv.size(); ++i) {
        if (rma_recv[i] != recv_vals[i]) {
            std::fprintf(stderr, "[%d] MISMATCH at %zu: PMPI=%d RMA=%d\n",
                         rank, i, recv_vals[i], rma_recv[i]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // -------- RMA winlock persistent --------
    alltoallv_rma_lock_init(packed_send.data(),
                            sendcounts.data(), sdispls.data(), MPI_INT,
                            rma_recv.data(),
                            recvcounts.data(), rdispls.data(), MPI_INT,
                            xcomm, xinfo, &req);

    MPI_Barrier(xcomm->global_comm);
    t0 = MPI_Wtime();
    for (int k = 0; k < n_iter; ++k) {
        MPIX_Start(req);
        MPIX_Wait(req, MPI_STATUS_IGNORE);
    }
    double rma_avg2 = (MPI_Wtime() - t0) / n_iter, rma_max2 = 0.0;
    MPI_Reduce(&rma_avg2, &rma_max2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("[RMA ] winlock persistent avg: %e s\n", rma_max2);
    }

    // correctness vs PMPI (again)
    if (rma_recv.size() != recv_vals.size()) {
        if (rank == 0) std::fprintf(stderr, "Size mismatch between PMPI and RMA_lock recv buffers\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (size_t i = 0; i < rma_recv.size(); ++i) {
        if (rma_recv[i] != recv_vals[i]) {
            std::fprintf(stderr, "[%d] MISMATCH at %zu: PMPI=%d RMA=%d\n",
                         rank, i, recv_vals[i], rma_recv[i]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPIX_Request_free(req);
    MPIX_Comm_win_free(xcomm);
    
    MPIX_Info_free(&xinfo);
    xinfo=nullptr;
   
     MPIX_Comm_free(&xcomm);
     xcomm=nullptr;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank=0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (argc < 2) {
        if (rank == 0) std::fprintf(stderr, "Usage: srun -n <P> ./sparse_pattern_with_rma <matrix.pm> [iters]\n");
        MPI_Finalize(); return 1;
    }
    const char* matrix = argv[1];
    int iters = (argc >= 3 ? std::max(1, std::atoi(argv[2])) : 10);
    test_matrix(matrix, iters);
    MPI_Finalize();
    return 0;
}
