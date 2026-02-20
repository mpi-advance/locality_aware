// Testing correctness and performance of OpenSHMEM Alltoallv persistent using SuiteSparse matrix communication patterns.
#include <shmem.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <time.h>

#include "osh_alltoallv.h"

#include "/usr/workspace/enamug/clean/GPU_locality_aware/locality_aware/src/tests/sparse_mat.hpp"
#include "/usr/workspace/enamug/clean/GPU_locality_aware/locality_aware/src/tests/par_binary_IO.hpp"

static inline double now_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void test_matrix(const char* filename, int n_iter)
{
    const int me   = shmem_my_pe();
    const int P    = shmem_n_pes();
    const int warmups = 10;

    //building SuiteSparse comm pattern (same as MPI version) 
    ParMat<int> A;
    readParMatrix(filename, A);
    form_comm(A);

   
    std::vector<double> send_vals(A.on_proc.n_rows);
    for (int i = 0; i < A.on_proc.n_rows; ++i) send_vals[i] = me * 1000.0 + (double)i;

    //  message index for send_comm procs
    std::vector<int> proc_pos(P, -1);
    for (int i = 0; i < A.send_comm.n_msgs; ++i)
        proc_pos[A.send_comm.procs[i]] = i;

    // Pack send buffer in dest order 0..P-1
    std::vector<double> packed_send(A.send_comm.size_msgs);
    int ctr = 0;
    for (int dest = 0; dest < P; ++dest) {
        int idx = proc_pos[dest];
        if (idx < 0) continue;
        int start = A.send_comm.ptr[idx];
        int end   = A.send_comm.ptr[idx + 1];
        for (int j = start; j < end; ++j) {
            int row = A.send_comm.idx[j];
            packed_send[ctr++] = send_vals[row];
        }
    }

    // counts & displacements (elements)
    std::vector<int> sendcounts(P, 0), recvcounts(P, 0);
    std::vector<int> sdispls(P + 1),    rdispls(P + 1);

    for (int i = 0; i < A.send_comm.n_msgs; ++i)
        sendcounts[A.send_comm.procs[i]] = A.send_comm.ptr[i + 1] - A.send_comm.ptr[i];

    for (int i = 0; i < A.recv_comm.n_msgs; ++i)
        recvcounts[A.recv_comm.procs[i]] = A.recv_comm.ptr[i + 1] - A.recv_comm.ptr[i];


    sdispls[0] = 0;
    rdispls[0] = 0;
    for (int p = 0; p < P; ++p) {
        sdispls[p + 1] = sdispls[p] + sendcounts[p];
        rdispls[p + 1] = rdispls[p] + recvcounts[p];
    }
//-------------------------------------------------------------------------------------------------------
   

    // Fixed-size symmetric buffers for metadata (counts/displs) and packed_send length (elements) 

    int* sendcounts_sym = (int*) shmem_malloc((size_t)P * sizeof(int));
    int* sdispls_sym    = (int*) shmem_malloc((size_t)P * sizeof(int));
    int* pack_elems_sym = (int*) shmem_malloc(sizeof(int));          

    // buffer to gather all packed_send lengths so we can compute global max (could also do a reduction, but got a bug in osh_alltoallv_reduce_max that way, so this is simpler)
    int* all_pack_elems = (int*) shmem_malloc((size_t)P * sizeof(int));
    int* max_pack_elems_sym = (int*) shmem_malloc(sizeof(int));

    if (!sendcounts_sym || !sdispls_sym || !pack_elems_sym ||
        !all_pack_elems || !max_pack_elems_sym) {
        if (me == 0) std::printf("ERROR: shmem_malloc failed for validation meta buffers\n");
        shmem_global_exit(1);
    }

    // Publish my local packed_send length
    *pack_elems_sym = (int)packed_send.size();

    // publish  sendcounts/sdispls in symmetric arrays
    for (int p = 0; p < P; ++p) {
        sendcounts_sym[p] = sendcounts[p];
        sdispls_sym[p]    = sdispls[p];   
    }

    shmem_barrier_all();

    // Gather all packed_send lengths @ PE 0 to compute global max (could also do a reduction, but got a bug in osh_alltoallv_reduce_max that way, so this is simpler)
    if (me == 0) all_pack_elems[0] = *pack_elems_sym;
    else         shmem_putmem(&all_pack_elems[me], pack_elems_sym, sizeof(int), 0);

    shmem_quiet();
    shmem_barrier_all();

  // Compute global max packed_send length and publish to all PEs so they can allocate symmetric buffer of the same size
    int max_pack_elems = 0;
    if (me == 0) {
        max_pack_elems = all_pack_elems[0];
        for (int p = 1; p < P; ++p)
            max_pack_elems = std::max(max_pack_elems, all_pack_elems[p]);

        *max_pack_elems_sym = max_pack_elems;
        for (int p = 1; p < P; ++p)
            shmem_putmem(max_pack_elems_sym, &max_pack_elems, sizeof(int), p);
    }

    shmem_quiet();
    shmem_barrier_all();

    max_pack_elems = *max_pack_elems_sym;

    // Now allocating the packed_send_sym (sendbuf) with same size everywhere (symmetry-safe)
    double* packed_send_sym =
        (double*) shmem_malloc((size_t)std::max(1, max_pack_elems) * sizeof(double));

    
    // copy my packed_send to packed_send_sym 
    if (max_pack_elems > 0) {
        
        if (!packed_send.empty()) {
            std::memcpy(packed_send_sym, packed_send.data(),
                        (size_t)packed_send.size() * sizeof(double));
        }
    }

    shmem_barrier_all(); // ensure all data is published before we start timing


//Getting the time for the init and free


     osh_a2avp_t* req = nullptr;
     void* recv_buf = nullptr;

     double t3= now_time();
    for(int i =0; i < n_iter; ++i)
    {

     
     osh_alltoallv_init((const void*)packed_send.data(),
        sendcounts.data(),
        sdispls.data(),          
        sizeof(double),
        &recv_buf,
        recvcounts.data(),
        rdispls.data(),         
        sizeof(double),
        &req
    );

     osh_a2avp_free(req);

    }

    double t4 = now_time();

   double avg_local_initfree = (t4 - t3) / (double)n_iter;


  /* ----next, Reduce to get MAX across all PEs (slowest process, this is similar to MPI_reduce ) ---- */


 typedef struct {
        double time;
       
    } bench_result_t;

bench_result_t mine1 = { avg_local_initfree };

bench_result_t *all = (bench_result_t*) shmem_malloc((size_t)P * sizeof(bench_result_t));
if (!all) shmem_global_exit(1);
// gather all results to PE 0 (safe because all is symmetric and same size on all PEs)
if (me == 0) all[0] = mine1;
else         shmem_putmem(&all[me], &mine1, sizeof(bench_result_t), 0);

shmem_quiet();
shmem_barrier_all();

if (me == 0) {
    double max_avg = all[0].time;
    for (int p = 1; p < P; ++p)
        if (all[p].time > max_avg) max_avg = all[p].time;

    printf("# INIT+FREE cost (avg per iter)\n");
    printf("# Columns: npes | avg_local_initfree(s) | max_global_initfree(s)\n");
    printf("%6d %20.6e %20.6e\n", P, all[0].time, max_avg);
    fflush(stdout);
}

shmem_barrier_all();



    // ---- OpenSHMEM persistent init ----


   
     osh_alltoallv_init((const void*)packed_send.data(),
        sendcounts.data(),
        sdispls.data(),          // sdispls[0..P-1] are starts
        sizeof(double),
        &recv_buf,
        recvcounts.data(),
        rdispls.data(),          // rdispls[0..P-1] are starts
        sizeof(double),
        &req
    );
   

    // ---- Timed iterations ----
    shmem_barrier_all();
     double t1= 0.0;

    for (int k = 0; k < warmups + n_iter; ++k) {
        if (k == warmups) t1 = now_time();
        req->start_function(req);
        req->wait_function(req);
    }
     double t2= now_time();
    double avg_local = (t2 - t1) / (double)n_iter;

    // validating correctness, receiver fetches expected blocks from each sender 
    
    double* recvbuf = (double*)req->recv_sym;
   // For each src, fetch their sendcount/disp, then fetch the expected block from their packed_send_sym, and validate against my recvbuf
    for (int src = 0; src < P; ++src)
     {
        int cnt = 0;
        int disp = 0;
        int src_pack_elems = 0;

        // getting count/cnt,disp from src 
        shmem_getmem(&cnt,  &sendcounts_sym[me], sizeof(int), src);
        shmem_getmem(&disp, &sdispls_sym[me],    sizeof(int), src);

        // geting src's real packed_send length so I don't read beyond it
        shmem_getmem(&src_pack_elems, pack_elems_sym, sizeof(int), src);
        shmem_quiet();

        if (cnt == 0) continue;

        // bounds check: disp+sendcount(cnt) must fit in src's packed_send
        if (disp < 0 || cnt < 0 || disp + cnt > src_pack_elems) {
           
            std::printf("[PE %d] FAIL(meta): src=%d disp=%d cnt=%d src_pack_elems=%d\n",
                        me, src, disp, cnt, src_pack_elems);
            std::fflush(stdout);
            shmem_global_exit(1);//will stop the test immediately on the meta failure;
        }
     // fetch expected block from src's packed_send_sym 
        std::vector<double> expected((size_t)cnt);
        shmem_getmem(expected.data(),
                     packed_send_sym + (size_t)disp,
                     (size_t)cnt * sizeof(double),
                     src);
        shmem_quiet();

        size_t base = (size_t)rdispls[src];
        // Validate recvbuf[base] agaisnt expected[0..cnt-1]
        for (int k = 0; k < cnt; ++k) {
            if (recvbuf[base + (size_t)k] != expected[(size_t)k]) {
                
                std::printf("[PE %d] FAIL: src=%d k=%d got=%g expected=%g (base=%zu cnt=%d disp=%d)\n",
                            me, src, k,
                            recvbuf[base + (size_t)k], expected[(size_t)k],
                            base, cnt, disp);
                std::fflush(stdout);
                
                shmem_global_exit(1);//will stop the test immediately on failure, then we can check the printed message to see what failed
            }
        }
        
    }

    //Gathering the time results 
   

    bench_result_t mine = { avg_local };

    bench_result_t* all_results =
        (bench_result_t*) shmem_malloc((size_t)P * sizeof(bench_result_t));
    if (!all_results) {
        if (me == 0) std::printf("alloc all_results failed\n");
        shmem_global_exit(1);
    }
// gather all results to PE 0 
    if (me == 0) all_results[0] = mine;
    else         shmem_putmem(&all_results[me], &mine, sizeof(bench_result_t), 0);

    shmem_quiet();
    shmem_barrier_all();

    // Now PE 0 computes global max time  and prints results
    if (me == 0) 
    {
        double max_time = all_results[0].time;
        

        for (int p = 1; p < P; ++p) {
            if (all_results[p].time > max_time) max_time = all_results[p].time;
           
        }

       
            std::printf("# OpenSHMEM Alltoallv Benchmark Results\n");
            std::printf("# Columns: npes | avg_local_time(s) | max_global_time(s) | correctness\n");
            std::printf("--------------------------------------------------------------------------\n");
            
        std::printf("%6d %20.6e %20.6e %14s\n",
                    P, all_results[0].time, max_time,
                    "PASS");
        std::fflush(stdout);
    }

    shmem_barrier_all();
    shmem_free(all);
    shmem_free(all_results);

    // ---- Cleanup ----
    osh_a2avp_free(req);

    shmem_barrier_all();
    shmem_free(packed_send_sym);

    shmem_free(sendcounts_sym);
    shmem_free(sdispls_sym);
    shmem_free(pack_elems_sym);
    shmem_free(all_pack_elems);
    shmem_free(max_pack_elems_sym);
}

int main(int argc, char** argv)
{
    shmem_init();
    const int me = shmem_my_pe();

    if (argc < 2) {
        if (me == 0) {
            std::fprintf(stderr,
                "Usage: oshrun -n <P> ./test_suitesparse_alltoallv_oshmem <matrix.pm> [iters]\n");
        }
        shmem_finalize();
        return 1;
    }

    const char* matrix = argv[1];
    int iters = (argc > 2) ? std::atoi(argv[2]) : 100;

    test_matrix(matrix, iters);

    shmem_finalize();
    return 0;
}




                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

