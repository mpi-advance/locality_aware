#include "osh_alltoallv.h"
#include <shmem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>//for fixed width integer types

// Function to calculate total recieve bytes(like in RMA)

static inline size_t total_recv_bytes(const int *counts, size_t elem_size, int npes) 
{
    size_t total_recv_bytes = 0;
    for (int i = 0; i < npes; i++) {
        total_recv_bytes += counts[i]*elem_size;
    }
    return total_recv_bytes;
}

//osh_alltoallv_init implementation using shmem
//Returns: 0 success, 1 bad args, 2 OOM(req), 3 OOM(recv_sym), 4 OOM(arrays)

int osh_alltoallv_init(const void      *sendbuf,
                       const int       *sendcounts,
                       const int       *sdispls,
                       size_t           send_size,  /* sizeof(send type) */
                       const int       *recvcounts,
                       const int       *rdispls,
                       size_t           recv_size,  /* sizeof(recv type) */
                       osh_a2avp_t    **request_ptr )
                       {

                         if (!request_ptr || !sendbuf || !sendcounts || !sdispls ||
        !recvcounts || !rdispls || send_size == 0 || recv_size == 0)
        return 1;

        const int me = shmem_my_pe();
        const int npes = shmem_n_pes();

        //  Allocate request and initialize it to 0

    osh_a2avp_t *req = (osh_a2avp_t*)malloc(sizeof(*req));
    if (!req) return 2;
    memset(req, 0, sizeof(*req));

       //  Filling basic meta and function info

    req->me        = me;
    req->npes      = npes;
    req->elem_size = recv_size;
    req->sendbuf   = sendbuf;
    req->start_function = osh_a2avp_start;
    req->wait_function  = osh_a2avp_wait;
      //recieve bytes i need for this pe
    req->recv_bytes_total = total_recv_bytes(recvcounts, recv_size, npes);

    shmem_barrier_all(); //barrier before symm allocation for the rdispls-all pes reach allocation together


    // Allocate symmetric memory for rdispls for all pes
    req->rdispls_sym = (size_t*)shmem_malloc(npes * sizeof(size_t));
    if (!req->rdispls_sym) {
        free(req);
        return 4;
    }

    shmem_barrier_all();

    // computing global max recv bytes so recv_buffer, recv_sym is symmetric(is the same size on all pes)
    size_t *gathered_recv_counts = (size_t*)shmem_malloc(npes * sizeof(size_t));
    size_t *max_recv_count = (size_t*)shmem_malloc(sizeof(size_t));
    if (!gathered_recv_counts || !max_recv_count) {
        shmem_free(req->rdispls_sym);
        free(req);
        return 4;
    }
    shmem_barrier_all();

    if (me == 0) gathered_recv_counts[0] = req->recv_bytes_total;
    else  shmem_putmem(&gathered_recv_counts[me], &req->recv_bytes_total, sizeof(size_t), 0);

    shmem_quiet(); //ensure all puts are done
    shmem_barrier_all();

    //compute max recv count(bytes), at this point we have all the recv_counts, the end max_recv_count[0] will have the max value

    size_t max_value = 0;
    if (me == 0) {
        max_value = gathered_recv_counts[0];
        for (int p = 1; p < npes; ++p) {
            if (gathered_recv_counts[p] > max_value) max_value = gathered_recv_counts[p];
        }

        *max_recv_count = max_value;
        for (int p = 1; p < npes; ++p) {
            shmem_putmem(max_recv_count, &max_value, sizeof(size_t), p);
        }
    }

    shmem_quiet();
    shmem_barrier_all();

    max_value = *max_recv_count;

    shmem_free(gathered_recv_counts);
    shmem_free(max_recv_count);

    //Allocate symmetric recv buffer with max recv count on all pes
    req->recv_sym = (size_t*)shmem_malloc(max_value);
    if (!req->recv_sym) {
        shmem_free(req->rdispls_sym);
        free(req);
        return 3;
    }


    // Allocate arrays for managing requests

    req->send_sizes   = (size_t*)malloc((size_t)npes * sizeof(size_t));
    req->recv_sizes   = (size_t*)malloc((size_t)npes * sizeof(size_t));
    req->sdispls_b    = (size_t*)malloc((size_t)npes * sizeof(size_t));
    req->put_displs_b = (size_t*)malloc((size_t)npes * sizeof(size_t));

    if (!req->send_sizes || !req->recv_sizes || !req->sdispls_b ||
        !req->put_displs_b) {
        free(req->send_sizes); free(req->recv_sizes);
        free(req->sdispls_b);  free(req->put_displs_b);
        shmem_free(req->recv_sym);
        shmem_free(req->rdispls_sym);
        free(req);
        return 4;
    }
    // Fill send_sizes, recv_sizes, sdispls_b
    for (int p = 0; p < npes; p++) {
        req->send_sizes[p]   = (size_t)sendcounts[p] * send_size;
        req->recv_sizes[p]   = (size_t)recvcounts[p] * recv_size;
        req->sdispls_b[p]    = (size_t)sdispls[p] * send_size;

    // publish displs in elements so peers will later multiply by recv_size, (make rdispls seen by other processes coz of sym memo) */
        req->rdispls_sym[p] = (size_t)rdispls[p];

    }
    shmem_barrier_all();// to ensure all pes have their rdipls in public rdispls_sym b4 PEs start to get rdispls after this point

    // Fetch remote rdispls (target offsets) (where *I* land on each PE d(destination pe buffer))

    for (int d = 0; d < npes; ++d) {
        size_t disp_me_on_d = 0; // displacement of me on d
        shmem_getmem(&disp_me_on_d, &req->rdispls_sym[me], sizeof(size_t), d);
        req->put_displs_b[d] = disp_me_on_d * req->elem_size; /* bytes */
    }

    shmem_quiet();
    shmem_barrier_all(); //ensure all PEs have their put_displs_b filled b4 starting the alltoallv

    *request_ptr = req;
    return 0;
}

int osh_a2avp_start(osh_a2avp_t *request) 
{
    if (!request) return 1;

    const int npes = request->npes;
    const void *sendbuf = request->sendbuf;

    const char *src_base = (const char*) request->sendbuf;
    char       *dst_base = (char*) request->recv_sym;

    for (int d = 0; d < request->npes; ++d) {
        size_t nbytes = request->send_sizes[d];
        if (nbytes == 0) continue;

        const char *src = src_base + request->sdispls_b[d];
        char       *dst = dst_base + request->put_displs_b[d];

         if (d == request->me) {
            // Local copy for self-send
            memcpy(dst, src, nbytes);
        } else {
            // Remote put for other PEs
            shmem_putmem(dst, src, nbytes, d);
        }
    }

    return 0;

}

int osh_a2avp_wait(osh_a2avp_t *request) 
{
    if (!request) return 0;

    shmem_quiet(); //ensure all puts are done
    shmem_barrier_all();

    return 0;
}

int osh_a2avp_free(osh_a2avp_t *request) 
{
    if (!request) return 0;

    if (request->recv_sym) shmem_free(request->recv_sym);
    if (request->rdispls_sym) shmem_free(request->rdispls_sym);

    free(request->send_sizes);
    free(request->recv_sizes);
    free(request->sdispls_b);
    free(request->put_displs_b);

    memset(request, 0, sizeof(*request));
    free(request);

    return 0;
}


