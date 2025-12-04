

#include <mpi.h>
#include <mpi-ext.h>// Open MPI 4.1.x persistent collectives
#include <vector>
#include <numeric>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "/g/g92/enamug/clean/GPU_locality_aware/locality_aware/src/tests/sparse_mat.hpp"
#include "/g/g92/enamug/clean/GPU_locality_aware/locality_aware/src/tests/par_binary_IO.hpp"
#include "/usr/workspace/enamug/clean/GPU_locality_aware/locality_aware/src/mpi_advance.h"


 void test_matrix(const char* filename, int n_iter)
{

    int rank, P; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &P);
    //P is num_procs
    // Load matrix 
    ParMat<int> A;

    //read suitesparse matrix
    readParMatrix(filename, A);

    //form comm pattern
    form_comm(A);

    
    std::vector<double> send_vals(A.on_proc.n_rows);
    for (int i = 0; i < A.on_proc.n_rows; ++i) send_vals[i] = rank*1000 + i;

    
     // Alltoallv_send_vals must be ordered (dest 0 to num_procs-1)
    std::vector<int> proc_pos(P, -1);
    for (int i = 0; i < A.send_comm.n_msgs; ++i)
        proc_pos[A.send_comm.procs[i]] = i;

    
    std::vector<double> packed_send(A.send_comm.size_msgs);//originally named "alltoallv_send_vals"-packing actual data to send
    int ctr = 0;
    for (int dest = 0; dest < P; ++dest) {
        int idx = proc_pos[dest];
        if (idx < 0) continue;  //skip ranks with no messages.
        int start = A.send_comm.ptr[idx];
        int end   = A.send_comm.ptr[idx+1];
        for (int j = start; j < end; ++j) {
            int row = A.send_comm.idx[j];
           
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
    std::vector<double> recv_vals(rdispls.back(), 0);
    MPI_Barrier(MPI_COMM_WORLD);//syc before timing

    int warmups = 10;
    double t0 = 0.0;
    
    for (int k = 0; k <warmups + n_iter; ++k) {
        if(k== warmups){

             t0 = MPI_Wtime();

        }
        MPI_Alltoallv(packed_send.data(),
                       sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                       recv_vals.data(),
                       recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }
    double pmpi_avg = (MPI_Wtime() - t0) / n_iter, pmpi_max = 0.0;
    MPI_Reduce(&pmpi_avg, &pmpi_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    
    if (rank == 0) {
       std::printf(" Alltoallv avg: %e\n", pmpi_max);
    }

    //---------Persistent Legacy Alltoallv_init

    
std::vector<double> recv_pers(rdispls.back(), 0);
MPI_Request req_pers = MPI_REQUEST_NULL;
/*
MPI_Barrier(MPI_COMM_WORLD);
 t0 = MPI_Wtime();
for (int k = 0; k < n_iter; ++k) {
    MPIX_Alltoallv_init(packed_send.data(),
                        sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                        recv_pers.data(),
                        recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                        MPI_COMM_WORLD, MPI_INFO_NULL, &req_pers);
    MPI_Request_free(&req_pers);
}
double pers_init_avg = (MPI_Wtime() - t0) / n_iter, pers_init_max = 0.0;
MPI_Reduce(&pers_init_avg, &pers_init_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
if (rank == 0) {
    std::printf("[PMPI] Alltoallv_init+free in loop avg: %e s\n", pers_init_max);
}

// === one time Persistent Alltoallv (Open MPI 4.1.x)=============
MPIX_Alltoallv_init(packed_send.data(),
                    sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                    recv_pers.data(),
                    recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                    MPI_COMM_WORLD, MPI_INFO_NULL, &req_pers);

//--------------real Persistent-----------------------------------
MPI_Barrier(MPI_COMM_WORLD);

    for (int k = 0; k <warmups + n_iter; ++k) {
        if(k== warmups){
            t0 = MPI_Wtime();

        }

    MPI_Start(&req_pers);
    MPI_Wait(&req_pers, MPI_STATUS_IGNORE);
}
double pers_avg = (MPI_Wtime() - t0) / n_iter, pers_max = 0.0;
MPI_Reduce(&pers_avg, &pers_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
if (rank == 0) {
    std::printf("[PMPI] Alltoallv persistent avg: %e s\n", pers_max);
}

 // correctness vs PMPI
    if (recv_pers.size() != recv_vals.size()) {
        if (rank == 0) std::fprintf(stderr, "Size mismatch between PMPI and  Persistent Alltoallv (Open MPI 4.1.x) buffers\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (size_t i = 0; i < recv_pers.size(); ++i) {
        if (recv_pers[i] != recv_vals[i]) {
            std::fprintf(stderr, "[%d] MISMATCH at %zu: PMPI=%d RMA=%d\n",
                         rank, i, recv_vals[i], recv_pers[i]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
*/
// i think i wont use req_pers again
MPI_Request_free(&req_pers);

//==============


    // ---RMA winfence persistent --------
    MPIX_Comm*    xcomm = nullptr;
    MPIX_Info*    xinfo = nullptr;
    MPIX_Request* req   = nullptr;

    MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);
    update_locality(xcomm, 4);  
    MPIX_Info_init(&xinfo);

    std::vector<double> rma_recv(rdispls.back(), 0);

    // measure init+free average 
    t0 = MPI_Wtime();
    for (int k = 0; k < n_iter; ++k) {
        alltoallv_rma_winfence_init(packed_send.data(),
                                    sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                    rma_recv.data(),
                                    recvcounts.data(), rdispls.data(), MPI_DOUBLE,
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
                                sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                rma_recv.data(),
                                recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                                xcomm, xinfo, &req);

    MPI_Barrier(xcomm->global_comm);
   
     for (int k = 0; k <warmups + n_iter; ++k) {
        if(k== warmups){

             t0 = MPI_Wtime();
        }
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
//==================winfence_han===============================================

t0 = MPI_Wtime();
    for (int k = 0; k < n_iter; ++k) {
        alltoallv_rma_winfence_init_han(packed_send.data(),
                                    sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                    rma_recv.data(),
                                    recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                                    xcomm, xinfo, &req);
        MPIX_Request_free(req);
    }
     rma_initavg = (MPI_Wtime() - t0) / n_iter; double rma_initmaxhan = 0.0;
    MPI_Reduce(&rma_initavg, &rma_initmaxhan, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("[RMA ] winfence init_han+free in loop avg: %e s\n", rma_initmaxhan);
    }


    


    // persistent request once
    alltoallv_rma_winfence_init_han(packed_send.data(),
                                sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                                rma_recv.data(),
                                recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                                xcomm, xinfo, &req);

    MPI_Barrier(xcomm->global_comm);
   
     for (int k = 0; k <warmups + n_iter; ++k) {
        if(k== warmups){

             t0 = MPI_Wtime();
        }
        MPIX_Start(req);
        MPIX_Wait(req, MPI_STATUS_IGNORE);
    }
    rma_avg = (MPI_Wtime() - t0) / n_iter; double rma_max_fencehan = 0.0;
    MPI_Reduce(&rma_avg, &rma_max_fencehan, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("[RMA ] winfence_han persistent avg: %e s\n", rma_max_fencehan);
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



//==================================Lock next==============================



  MPI_Barrier(xcomm->global_comm);
   double tlnit = MPI_Wtime();
    // This is for accuracy

    for (int k = 0; k < n_iter; k++) {
        // printf("rank:%d\n,k =%d\n",rank,k);
        // printf("****a\n");
         alltoallv_rma_lock_init(packed_send.data(),
                            sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                            rma_recv.data(),
                            recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                            xcomm, xinfo, &req);
        // printf("****b\n");
        MPIX_Request_free(req);
        MPIX_Comm_win_free(xcomm);
        // printf("****c\n");
    }

    double rma_intlock_final = (MPI_Wtime() - tlnit) / n_iter, rma_intlock_final_max=0.0;

    // RMA 
    MPI_Reduce(&rma_intlock_final, &rma_intlock_final_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("[RMA ] winlock init+free in loop avg: %e seconds\n", rma_intlock_final_max);
       
    }


    // -------- RMA winlock persistent --------
    alltoallv_rma_lock_init(packed_send.data(),
                            sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                            rma_recv.data(),
                            recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                            xcomm, xinfo, &req);

    MPI_Barrier(xcomm->global_comm);
    //t0 = MPI_Wtime();
   for (int k = 0; k <warmups + n_iter; ++k) {
        if(k== warmups){
        t0 = MPI_Wtime();
        }

        MPIX_Start(req);
        MPIX_Wait(req, MPI_STATUS_IGNORE);
        
    }
    double rma_avg2 = (MPI_Wtime() - t0) / n_iter, rma_max2 = 0.0;
    MPI_Reduce(&rma_avg2, &rma_max2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("[RMA ] winlock persistent avg: %e s\n", rma_max2);
    }

    // correctness vs PMPI 
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
//==========================Lock-Han next=======================================================


MPI_Barrier(xcomm->global_comm);
    tlnit = MPI_Wtime();
    // This is for accuracy

    for (int k = 0; k < n_iter; k++) {
        // printf("rank:%d\n,k =%d\n",rank,k);
        // printf("****a\n");
        // printf("[rank %d] LOOP winlock_han init+free, iter %d: calling alltoallv_rma_lock_init_han\n",
         //  rank, k);
    //fflush(stdout);

         alltoallv_rma_lock_init_han(packed_send.data(),
                            sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                            rma_recv.data(),
                            recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                            xcomm, xinfo, &req);
        //printf("[rank %d] LOOP winlock_han init+free B, iter %d: MPIX_Request_free\n",
         //  rank, k);
    //fflush(stdout);
        MPIX_Request_free(req);
        //printf("[rank %d] LOOP winlock_han init+free C, iter %d: MPIX_Request_free\n",
          // rank, k);
    //fflush(stdout);
        MPIX_Comm_win_free(xcomm);
        // printf("****c\n");
        //printf("[rank %d] LOOP winlock_han init+free D, iter %d: MPIX_Request_free\n",
          // rank, k);
    //fflush(stdout);
    }

     rma_intlock_final = (MPI_Wtime() - tlnit) / n_iter; double rma_intlock_finali_max=0.0;

    // RMA 
    MPI_Reduce(&rma_intlock_final, &rma_intlock_finali_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("[RMA ] winlock init_han + free in loop avg: %e seconds\n", rma_intlock_finali_max);
       
    }


    // -------- RMA winlock han persistent --------
    alltoallv_rma_lock_init_han(packed_send.data(),
                            sendcounts.data(), sdispls.data(), MPI_DOUBLE,
                            rma_recv.data(),
                            recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                            xcomm, xinfo, &req);

    MPI_Barrier(xcomm->global_comm);
    //t0 = MPI_Wtime();
   for (int k = 0; k <warmups + n_iter; ++k) {
        if(k== warmups){
        t0 = MPI_Wtime();
        }
      // printf("****start 1 %d\n",rank);
        MPIX_Start(req);
       //printf("****start 2 %d\n",rank);
        MPIX_Wait(req, MPI_STATUS_IGNORE);
        //printf("****leaving start + wait %d\n", rank);
        
    }
     rma_avg2 = (MPI_Wtime() - t0) / n_iter; double rma_max3 = 0.0;
    MPI_Reduce(&rma_avg2, &rma_max3, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("[RMA ] winlock_han persistent avg: %e s\n", rma_max3);
    }

    // correctness vs PMPI 
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
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (argc < 2) {
        if (rank == 0) std::fprintf(stderr, "Usage: srun -n <P> ./sparse_pattern_with_rma <matrix.pm> [iters]\n");
        MPI_Finalize(); return 1;
    }
    const char* matrix = argv[1];
    int iters = argv[2];
    
    test_matrix(matrix, iters);
    MPI_Finalize();
    return 0;
}

