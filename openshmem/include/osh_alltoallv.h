
#ifndef OSH_ALLTOALLV_H
#define OSH_ALLTOALLV_H



#include <stddef.h>   /* for size_t */
#include <shmem.h>    /* OpenSHMEM API */

#ifdef __cplusplus
extern "C" {
#endif


/* ------------------------------
 *   Request structure
 * ------------------------------ */

typedef struct osh_a2avp {
    int      me;                // my PE rank
    int      npes;              // total number of PEs
    size_t   elem_size;         // size of one receive element in bytes

    const void *sendbuf;        // user's send buffer
    void       *recv_sym;       // symmetric receive buffer ("window")
    size_t      recv_bytes_total; // total bytes allocated symmetrically
    size_t      recv_bytes_alloc;

    /* Per-peer metadata */
    size_t   *send_sizes;       // bytes to send to peer i
    size_t   *recv_sizes;       // bytes expected from peer i
    size_t   *sdispls_b;        // byte displacements of local sendbuf
    size_t   *put_displs_b;     // byte displacements in target recv_sym

    /* Symmetric recv displacements (published by each PE) */
    size_t   *rdispls_sym;      //shmem_getmem to get the targets offset

    /* Persistent operations */
    int (*start_function)(struct osh_a2avp *r);//puts or local copies
    int (*wait_function) (struct osh_a2avp *r);// quiet and barrier to complete all puts and 4 global synchronisation
} osh_a2avp_t;


/*
 * Initialize a persistent Alltoallv operation.
 * Allocates all necessary symmetric memory in the .c.
 */
int osh_alltoallv_init(const void      *sendbuf,
                                const int       *sendcounts,
                                const int       *sdispls,
                                size_t          send_size,
                                const int       *recvcounts,
                                const int       *rdispls,
                                size_t          recv_size,
                                osh_a2avp_t    **request_ptr);



/*
 *
 * Performs the actual puts or local copies.
 */
int osh_a2avp_start(osh_a2avp_t *request);


/*
 * Complete the operation: using quiet and barrier.
 */
int osh_a2avp_wait(osh_a2avp_t *request);

/*
 * Free all dynamically and symmetrically allocated memory.
 * Frees the request structure as well.
 */
int osh_a2avp_free(osh_a2avp_t *request);
#ifdef __cplusplus
} // extern "C"
#endif

#endif



