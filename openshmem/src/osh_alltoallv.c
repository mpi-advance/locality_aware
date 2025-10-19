
#include "osh_alltoallv.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>


struct osh_a2avp {
    int        me, npes;
    size_t     elem_size;

    
    void      *recv_sym;
    size_t     recv_bytes_total;
    int        recv_owned;

    
    void     **dst_ptr;    
    size_t    *nbytes;    
    ptrdiff_t *sdispls_e;  

    
    ptrdiff_t *rdispls_sym;
};

/* --Helpers --*/
static size_t sum_bytes(const size_t *cnts, size_t elem_size, int npes) {
    size_t s = 0;
    for (int i = 0; i < npes; ++i) s += cnts[i];
    return s * elem_size;
}


/*  PERSISTENT OSHMEM  API  STARTS NOW!  */

int osh_a2avp_init(osh_a2avp_t **req_out,
                   const void    *sendbuf,
                   const size_t  *sendcounts,
                   const ptrdiff_t *sdispls_elems,
                   size_t          elem_size,
                   void          **recvbuf_sym_inout,
                   const size_t  *recvcounts,
                   const ptrdiff_t *rdispls_elems)
{
    if (!req_out || !sendcounts || !sdispls_elems || !recvcounts || !rdispls_elems || elem_size == 0)
        return 1;

    shmem_barrier_all(); 

    osh_a2avp_t *r = (osh_a2avp_t*) calloc(1, sizeof(*r));
    if (!r) return 2;

    r->me        = shmem_my_pe();
    r->npes      = shmem_n_pes();
    r->elem_size = elem_size;

    /* Setup symmetric recv buffer (user-provided or lib-allocated) */
    r->recv_bytes_total = sum_bytes(recvcounts, elem_size, r->npes);
    if (recvbuf_sym_inout && *recvbuf_sym_inout) {
        r->recv_sym  = *recvbuf_sym_inout;
        r->recv_owned = 0;
    } else {
        r->recv_sym = shmem_malloc(r->recv_bytes_total);
        if (!r->recv_sym) { free(r); return 3; }
        r->recv_owned = 1;
        if (recvbuf_sym_inout) *recvbuf_sym_inout = r->recv_sym;
    }

    /* Allocate  arrays */
    r->dst_ptr   = (void**)     malloc(sizeof(void*)    * (size_t)r->npes);
    r->nbytes    = (size_t*)    malloc(sizeof(size_t)   * (size_t)r->npes);
    r->sdispls_e = (ptrdiff_t*) malloc(sizeof(ptrdiff_t)* (size_t)r->npes);
    if (!r->dst_ptr || !r->nbytes || !r->sdispls_e) {
        free(r->dst_ptr); free(r->nbytes); free(r->sdispls_e);
        if (r->recv_owned && r->recv_sym) shmem_free(r->recv_sym);
        free(r);
        return 4;
    }

    
    memcpy(r->sdispls_e, sdispls_elems, sizeof(ptrdiff_t) * (size_t)r->npes);

    
    for (int d = 0; d < r->npes; ++d) r->nbytes[d] = sendcounts[d] * elem_size;

    
    r->rdispls_sym = (ptrdiff_t*) shmem_malloc((size_t)r->npes * sizeof(ptrdiff_t));
    if (!r->rdispls_sym) {
        free(r->dst_ptr); free(r->nbytes); free(r->sdispls_e);
        if (r->recv_owned && r->recv_sym) shmem_free(r->recv_sym);
        free(r);
        return 5;
    }
    memcpy(r->rdispls_sym, rdispls_elems, (size_t)r->npes * sizeof(ptrdiff_t));

   
    shmem_barrier_all();//wait

    
    for (int d = 0; d < r->npes; ++d) {
        ptrdiff_t disp_me_on_d = 0;

        /*  The remote address is symmetric for recv 
            */
        const void *remote_addr =
            (const void*)((const char*)r->rdispls_sym + (size_t)r->me * sizeof(ptrdiff_t));

        shmem_getmem(&disp_me_on_d, remote_addr, sizeof(ptrdiff_t), d);

        r->dst_ptr[d] = (void*)((char*)r->recv_sym + (size_t)disp_me_on_d * r->elem_size);
    }

    shmem_barrier_all();
    *req_out = r;
    return 0;
}

int osh_a2avp_start(osh_a2avp_t *r, const void *sendbuf_current)
{
    if (!r || !sendbuf_current) return 1;

    const int npes = r->npes;
    const size_t esz = r->elem_size;

    for (int d = 0; d < npes; ++d) {
        const size_t nb = r->nbytes[d];
        if (nb == 0) continue;

        const void *src = (const void*)((const char*)sendbuf_current + (size_t)r->sdispls_e[d] * esz);
        void *dst_remote = r->dst_ptr[d];

        if (d == r->me) {
            memcpy(dst_remote, src, nb);
        } else {
            shmem_putmem_nbi(dst_remote, src, nb, d);
        }
    }
    return 0;
}

int osh_a2avp_wait(osh_a2avp_t *r)
{
    (void)r;
    shmem_quiet();
    return 0;
}

void osh_a2avp_free(osh_a2avp_t *r)
{
    if (!r) return;
    if (r->dst_ptr)     free(r->dst_ptr);
    if (r->nbytes)      free(r->nbytes);
    if (r->sdispls_e)   free(r->sdispls_e);
    if (r->rdispls_sym) shmem_free(r->rdispls_sym);
    if (r->recv_owned && r->recv_sym) shmem_free(r->recv_sym);
    free(r);
}

void*  osh_a2avp_recvbuf(osh_a2avp_t *r)   { return r ? r->recv_sym        : NULL; }
size_t osh_a2avp_recvbytes(osh_a2avp_t *r) { return r ? r->recv_bytes_total : 0;   }

