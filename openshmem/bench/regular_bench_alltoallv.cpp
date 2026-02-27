
#include <shmem.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <time.h>

#include "osh_alltoallv.h"

static inline double now_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

//  gather max time at PE0 and print one line of results with a header block 
static void gather_max_time_and_print_status(const char* header_block,
                                             int me, int npes,
                                             double local_time,
                                             const char* status)
{
    typedef struct { double time; } bench_result_t;

    bench_result_t mine = { local_time };
    bench_result_t* all = (bench_result_t*) shmem_malloc((size_t)npes * sizeof(bench_result_t));
    if (!all) shmem_global_exit(1);

    shmem_barrier_all();
    if (me == 0) all[0] = mine;
    else         shmem_putmem(&all[me], &mine, sizeof(bench_result_t), 0);

    shmem_quiet();
    shmem_barrier_all();

    if (me == 0) {
        double max_time = all[0].time;
        for (int p = 1; p < npes; ++p) {
            if (all[p].time > max_time) max_time = all[p].time;
        }

        std::printf("%s\n", header_block);
        std::printf("%6d %20.6e %20.6e %14s\n", npes, all[0].time, max_time, status);
        std::fflush(stdout);
    }

    shmem_barrier_all();
    shmem_free(all);
}

int main(int argc, char **argv)
{
    shmem_init();
    const int me   = shmem_my_pe();
    const int npes = shmem_n_pes();

    const size_t elem_size = sizeof(double);

    int T =262144;   // elements per destination
    int iters = 100;   // number of timed iterations (after warmups)
    int warmups = 10;

  
    std::vector<int> sendcounts(npes, T);
    std::vector<int> recvcounts(npes, T);
    std::vector<int> sdispls(npes);
    std::vector<int> rdispls(npes);

    for (int p = 0; p < npes; ++p) {
        sdispls[p] = p * T;
        rdispls[p] = p * T;
    }

    const int send_total_elems = npes * T;
    const int recv_total_elems = npes * T;

    
    std::vector<double> sendbuf((size_t)send_total_elems);
    for (int i = 0; i < send_total_elems; ++i) {
        sendbuf[i] = me;
    }

    // Print message size summary just for results 
    if (me == 0) {
        const size_t bytes_per_dest = (size_t)T * elem_size;
        const size_t total_send_bytes_per_proc = (size_t)npes * bytes_per_dest;
        std::printf("# Message size summary (REGULAR)\n");
        std::printf("# per-destination bytes: %zu\n", bytes_per_dest);
        std::printf("# total send bytes per process: %zu\n", total_send_bytes_per_proc);
        std::fflush(stdout);
    }

    osh_a2avp_t* req = nullptr;
    void* recv_buf = nullptr;
    double t3 = now_time();
    
    for (int i = 0; i < iters; ++i) {
        

         osh_alltoallv_init((const void*)sendbuf.data(),
                                    sendcounts.data(),
                                    sdispls.data(),
                                    elem_size,
                                    &recv_buf,            // outward
                                    recvcounts.data(),
                                    rdispls.data(),
                                    elem_size,
                                    &req);
        

        osh_a2avp_free(req);
    }
    double t4 = now_time();
    double avg_local_initfree = (t4 - t3) / (double)iters;

    gather_max_time_and_print_status(
        "# INIT+FREE cost (avg per iter)\n"
        "# Columns: npes | avg_local_initfree(s) | max_global_initfree(s) | correctness\n"
        "--------------------------------------------------------------------------",
        me, npes, avg_local_initfree, "N/A");

    
    // one persistent init for the actual timed start+wait 
   

     osh_alltoallv_init((const void*)sendbuf.data(),
                                sendcounts.data(),
                                sdispls.data(),
                                elem_size,
                                &recv_buf,              // outward
                                recvcounts.data(),
                                rdispls.data(),
                                elem_size,
                                &req);
    
    

    
     recv_buf = req->recv_sym;

    
    // Timed iterations; start+wait
    
    shmem_barrier_all();

    double t0 = 0.0;
    for (int k = 0; k < warmups + iters; ++k) {
        if (k == warmups) t0 = now_time();
        req->start_function(req);
        req->wait_function(req);
    }
    double avg_local = (now_time() - t0) / (double)iters;


    // ------------------------------------------------------------
    // Validation ( no packed_send_sym needed like in the irregular case, because this is a regular pattern and recvbuf is already in the expected order, so we can directly check recv_buf against expected values)
    //
    // With regular pattern:
    //  From each src, I receive T elements
    //  They are placed at recvbuf[rdispls[src] 
    //  Each element should equal src becoz we filled sendbuf with src
    // ------------------------------------------------------------
    double* recvbuf = (double*)recv_buf;

    for (int src = 0; src < npes; ++src) {
        const int base = rdispls[src];
        for (int j = 0; j < T; ++j) {
            const double got = recvbuf[(size_t)(base + j)];
            const double expected = (double)src;
            if (got != expected) {
                printf("[PE %d] FAIL: src=%d j=%d got=%g expected=%g (base=%d T=%d)\n",
                            me, src, j, got, expected, base, T);
                std::fflush(stdout);
                shmem_global_exit(1);
            }
        }
    }

    // If we reached here, correctness passed on this PE.
    gather_max_time_and_print_status(
        "# OpenSHMEM Alltoallv Benchmark Results (REGULAR)\n"
        "# Columns: npes | avg_local_time(s) | max_global_time(s) | correctness\n"
        "--------------------------------------------------------------------------",
        me, npes, avg_local, "PASS");

    
    // Cleanup
    
    osh_a2avp_free(req);//recv_buf is part of req->recv_sym, so it will be freed in osh_a2avp_free

    shmem_finalize();
    return 0;

    }



