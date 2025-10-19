
#include <shmem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "osh_alltoallv.h"  

static inline double now_time() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct { double init_t, p_avg; } Row;

int main(int argc, char **argv) {
    shmem_init();
    int me = shmem_my_pe(), npes = shmem_n_pes();

    const size_t elem_size = sizeof(double);
    int iters  = 128;
    int T      = 100;      // fixed
    if (argc > 1) T     = atoi(argv[1]);  // per-destination elements
    if (argc > 2) iters = atoi(argv[2]);

    // Building counts/displacements 
    size_t   *sendcounts = (size_t*)   malloc((size_t)npes * sizeof(size_t));
    size_t   *recvcounts = (size_t*)   malloc((size_t)npes * sizeof(size_t));
    ptrdiff_t *sdispls   = (ptrdiff_t*)malloc((size_t)npes * sizeof(ptrdiff_t));
    ptrdiff_t *rdispls   = (ptrdiff_t*)malloc((size_t)npes * sizeof(ptrdiff_t));
    if (!sendcounts || !recvcounts || !sdispls || !rdispls) {
        if (me == 0) fprintf(stderr, "failed\n");
        shmem_finalize();
        return 1;
    }

    // same counts T elements to/from every PE, displacements = prefix sums 
    size_t send_total_elems = 0, recv_total_elems = 0;
    for (int p = 0, off = 0; p < npes; ++p) {
        sendcounts[p] = (size_t)T;
        sdispls[p]    = (ptrdiff_t)off;
        off          += T;//where should the next block begin
    }
    send_total_elems = (size_t)npes * (size_t)T;

    for (int p = 0, off = 0; p < npes; ++p) {
        recvcounts[p] = (size_t)T;          // expecting T from source p
        rdispls[p]    = (ptrdiff_t)off;
        off          += T;
    }
    recv_total_elems = (size_t)npes * (size_t)T;

    // Message-sizes-same
    const size_t bytes_per_dest = (size_t)T * elem_size;
    const size_t total_send_bytes_per_proc = (size_t)npes * bytes_per_dest;
    if (me == 0) {
        printf("# Message size summary (REGULAR 4 NOW):\n");
        printf("  Per-destination bytes: %zu\n", bytes_per_dest);
        printf("  Total send bytes per process: %zu\n", total_send_bytes_per_proc);
    }

    //  Allocating and filling send buffer  
   double *sendbuf = malloc(send_total_elems * sizeof(double));

   if (!sendbuf) {
    if (me == 0) fprintf(stderr, "malloc(sendbuf) failed\n");
    free(sendcounts); free(recvcounts); free(sdispls); free(rdispls);
    shmem_finalize();
    return 1;
   }
   //filling the sendbuf
for (int dest = 0; dest < npes; ++dest) {
    for (int i = 0; i < T; ++i) {
        size_t idx = sdispls[dest] + i;
        sendbuf[idx] = 1000.0 * me + 10.0 * dest + i;//can also make it same
    }
}
if (me == 0) printf("Calling osh_a2avp_init...\n"); fflush(stdout);//checking 1,2 :)
    //lib allocates symmetric recv if NULL 
    void *recv_sym = NULL;

    double t0 = now_time();
    osh_a2avp_t *req = NULL;

    osh_a2avp_init(&req,
                        sendbuf, sendcounts, sdispls, elem_size,
                        &recv_sym, recvcounts, rdispls);

    double init_time = now_time() - t0;

    // Timing oshmem_int periteration (start + wait) 
    shmem_barrier_all();

    double sum_p = 0.0;

    for (int it = 0; it < iters; ++it) {
        double a = now_time();
        osh_a2avp_start(req, sendbuf);
        osh_a2avp_wait(req);
        double b = now_time();
        sum_p += (b - a);
    }

    double p_avg = sum_p / iters;

    shmem_barrier_all();

    // Gathering to PE 0 and printing summary 
    Row *rows = (Row*) shmem_malloc(sizeof(Row) * (size_t)npes);
    if (!rows) {
        if (me == 0) fprintf(stderr, "shmem_malloc(rows) failed\n");
        osh_a2avp_free(req);
        free(sendbuf);
        free(sendcounts); free(recvcounts); free(sdispls); free(rdispls);
        shmem_finalize();
        return 3;
    }

    Row mine = (Row){ init_time, p_avg };

    shmem_barrier_all();
    if (me == 0) rows[0] = mine;
    else         shmem_putmem_nbi(&rows[me], &mine, sizeof(Row), 0);
    shmem_quiet();
    shmem_barrier_all();

    if (me == 0) {
        double max_init = rows[0].init_t;  int pe_init = 0;
        double max_pavg = rows[0].p_avg;   int pe_pavg = 0;
        double mean_p   = rows[0].p_avg;

        for (int p = 1; p < npes; ++p) {
            if (rows[p].init_t > max_init) { max_init = rows[p].init_t; pe_init = p; }
            if (rows[p].p_avg  > max_pavg) { max_pavg = rows[p].p_avg; pe_pavg = p; }
            mean_p += rows[p].p_avg;
        }
        mean_p /= npes;

        printf("# npes=%d, iters=%d, elem_size=%zu bytes\n", npes, iters, elem_size);
        printf("Persistent avg/iter: mean=%.9f s | max=%.9f s (PE %d)\n", mean_p, max_pavg, pe_pavg);
        printf("MAX init time       = %.9f s (PE %d)\n", max_init, pe_init);
        printf("Amortized MAX init/iter = %.9e s\n", max_init / (double)iters);
    }

    shmem_barrier_all();

    //  Cleaning up 
    shmem_free(rows);
    osh_a2avp_free(req);
    free(sendbuf);
    free(sendcounts); free(recvcounts); free(sdispls); free(rdispls);
    shmem_finalize();
    return 0;
}
