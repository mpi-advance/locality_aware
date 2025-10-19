#ifndef OSH_ALLTOALLV_PERSIST_H
#define OSH_ALLTOALLV_PERSIST_H


#include <stddef.h>
#include <shmem.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct osh_a2avp osh_a2avp_t;

// Creating persistent request. returns 0.
int osh_a2avp_init(osh_a2avp_t **req_out,
                   const void    *sendbuf,           // local
                   const size_t  *sendcounts,        // len=npes
                   const ptrdiff_t *sdispls_elems,   // len=npes
                   size_t          elem_size,
                   void          **recvbuf_sym_inout,// if *NULL -> lib allocs; else must be symmetric
                   const size_t  *recvcounts,        // len=npes
                   const ptrdiff_t *rdispls_elems    // len=npes (displacements for sources 0-npes-1 on THIS PE)
                   );

// issueing all nonblocking puts
int osh_a2avp_start(osh_a2avp_t *req, const void *sendbuf_current);


int osh_a2avp_wait(osh_a2avp_t *req);


void osh_a2avp_free(osh_a2avp_t *req);

// Accessors
void*  osh_a2avp_recvbuf(osh_a2avp_t *req);
size_t osh_a2avp_recvbytes(osh_a2avp_t *req);






#ifdef __cplusplus
}
#endif

#endif 
