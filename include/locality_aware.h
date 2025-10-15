#ifndef MPI_ADVANCE_H
#define MPI_ADVANCE_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Objects offered by this header*/
typedef struct _MPIL_Comm MPIL_Comm;
typedef struct _MPIL_Info MPIL_Info;
typedef struct _MPIL_Topo MPIL_Topo;
typedef struct _MPIL_Request MPIL_Request;

typedef int (*mpil_start_ftn)(MPIL_Request* request);
typedef int (*mpil_wait_ftn)(MPIL_Request* request, MPI_Status* status);

//Basic override for MPI_Alltoallv
int MPI_Alltoallv(const void* sendbuf,
                  const int sendcounts[],
                  const int sdispls[],
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcounts[],
                  const int rdispls[],
                  MPI_Datatype recvtype,
                  MPI_Comm comm); 

//Functions to control various versions of the MPIL_Comm object---------------------
int MPIL_Comm_init(MPIL_Comm** xcomm_ptr, MPI_Comm global_comm);
int MPIL_Comm_free(MPIL_Comm** xcomm_ptr);

int MPIL_Comm_topo_init(MPIL_Comm* xcomm);
int MPIL_Comm_topo_free(MPIL_Comm* xcomm);

int MPIL_Comm_leader_init(MPIL_Comm* xcomm, int procs_per_leader);
int MPIL_Comm_leader_free(MPIL_Comm* xcomm);

int MPIL_Comm_win_init(MPIL_Comm* xcomm, int bytes, int type_bytes);
int MPIL_Comm_win_free(MPIL_Comm* xcomm);

int MPIL_Comm_device_init(MPIL_Comm* xcomm);
int MPIL_Comm_device_free(MPIL_Comm* xcomm);

int MPIL_Comm_req_resize(MPIL_Comm* xcomm, int n);

/** @brief get current tag and increment tag in the comm.**/
int MPIL_Comm_tag(MPIL_Comm* comm, int* tag);

void update_locality(MPIL_Comm* xcomm, int ppn);

//Functions to initialize and free the MPI_Info object---------------------------------
int MPIL_Info_init(MPIL_Info** info);
int MPIL_Info_free(MPIL_Info** info);

//Functions to control the MPIL_Topo object---------------------------------------------
int MPIL_Topo_init(int indegree,
                   const int sources[],
                   const int sourceweights[],
                   int outdegree,
                   const int destinations[],
                   const int destweights[],
                   MPIL_Info* info,
                   MPIL_Topo** mpil_topo_ptr);
int MPIL_Topo_from_neighbor_comm(MPIL_Comm* comm, MPIL_Topo** mpil_topo_ptr);
int MPIL_Topo_free(MPIL_Topo** topo);

//Function to control the MPIL_Request object -----------------------------------------------------
/*// Starting locality-aware requests
// 1. Start Local_L
// 2. Start and wait for local_S
// 3. Start global  */
int MPIL_Start(MPIL_Request* request);
/*Wait for locality-aware requests
// 1. Wait for global
// 2. Start and wait for local_R
// 3. Wait for local_L  */
int MPIL_Wait(MPIL_Request* request, MPI_Status* status);
int MPIL_Request_free(MPIL_Request** request);


//Basic collective alltoall function--------------------------------------------------------------
enum AlltoallMethod
{
    ALLTOALL_PAIRWISE,
    ALLTOALL_NONBLOCKING,
    ALLTOALL_HIERARCHICAL_PAIRWISE,
    ALLTOALL_HIERARCHICAL_NONBLOCKING,
    ALLTOALL_MULTILEADER_PAIRWISE,
    ALLTOALL_MULTILEADER_NONBLOCKING,
    ALLTOALL_NODE_AWARE_PAIRWISE,
    ALLTOALL_NODE_AWARE_NONBLOCKING,
    ALLTOALL_LOCALITY_AWARE_PAIRWISE,
    ALLTOALL_LOCALITY_AWARE_NONBLOCKING,
    ALLTOALL_MULTILEADER_LOCALITY_PAIRWISE,
    ALLTOALL_MULTILEADER_LOCALITY_NONBLOCKING,
    ALLTOALL_PMPI
};
extern enum AlltoallMethod mpil_alltoall_implementation;
void MPIL_set_alltoall_algorithm(enum AlltoallMethod algorithm);

int MPIL_Alltoall(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* comm);


//Basic collective alltoallv function---------------------------------------------------------------------
enum AlltoallvMethod
{
    ALLTOALLV_PAIRWISE,
    ALLTOALLV_NONBLOCKING,
    ALLTOALLV_BATCH,
    ALLTOALLV_BATCH_ASYNC,
    ALLTOALLV_PMPI
};
extern enum AlltoallvMethod mpil_alltoallv_implementation;
void MPIL_set_alltoallv_algorithm(enum AlltoallvMethod algorithm);
int MPIL_Alltoallv(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm);

//Neighboor alltoallv functions --------------------------------------------------------------------------
int MPIL_Dist_graph_create_adjacent(MPI_Comm comm_old,
                                    int indegree,
                                    const int sources[],
                                    const int sourceweights[],
                                    int outdegree,
                                    const int destinations[],
                                    const int destweights[],
                                    MPIL_Info* info,
                                    int reorder,
                                    MPIL_Comm** comm_dist_graph_ptr);

enum NeighborAlltoallvMethod
{
    NEIGHBOR_ALLTOALLV_STANDARD,
    NEIGHBOR_ALLTOALLV_LOCALITY
};
extern enum NeighborAlltoallvMethod mpil_neighbor_alltoallv_implementation;
void MPIL_set_alltoall_neighbor_alogorithm(enum NeighborAlltoallvMethod algorithm);

int MPIL_Neighbor_alltoallv(const void* sendbuf,
                            const int sendcounts[],
                            const int sdispls[],
                            MPI_Datatype sendtype,
                            void* recvbuf,
                            const int recvcounts[],
                            const int rdispls[],
                            MPI_Datatype recvtype,
                            MPIL_Comm* comm);
int MPIL_Neighbor_alltoallv_topo(const void* sendbuf,
                                 const int sendcounts[],
                                 const int sdispls[],
                                 MPI_Datatype sendtype,
                                 void* recvbuf,
                                 const int recvcounts[],
                                 const int rdispls[],
                                 MPI_Datatype recvtype,
                                 MPIL_Topo* topo,
                                 MPIL_Comm* comm);

//Persisent Neighboorhood alltoallv functions-------------------------------------------------------------
enum NeighborAlltoallvInitMethod
{
    NEIGHBOR_ALLTOALLV_INIT_STANDARD,
    NEIGHBOR_ALLTOALLV_INIT_LOCALITY
};
extern enum NeighborAlltoallvInitMethod mpil_neighbor_alltoallv_init_implementation;
void MPIL_set_alltoallv_neighbor_init_alogorithm(enum NeighborAlltoallvInitMethod algorithm);

int MPIL_Neighbor_alltoallv_init(const void* sendbuf,
                                 const int sendcounts[],
                                 const int sdispls[],
                                 MPI_Datatype sendtype,
                                 void* recvbuf,
                                 const int recvcounts[],
                                 const int rdispls[],
                                 MPI_Datatype recvtype,
                                 MPIL_Comm* comm,
                                 MPIL_Info* info,
                                 MPIL_Request** request_ptr);
int MPIL_Neighbor_alltoallv_init_ext(const void* sendbuf,
                                     const int sendcounts[],
                                     const int sdispls[],
                                     const long global_sindices[],
                                     MPI_Datatype sendtype,
                                     void* recvbuf,
                                     const int recvcounts[],
                                     const int rdispls[],
                                     const long global_rindices[],
                                     MPI_Datatype recvtype,
                                     MPIL_Comm* comm,
                                     MPIL_Info* info,
                                     MPIL_Request** request_ptr);
int MPIL_Neighbor_alltoallv_init_topo(const void* sendbuf,
                                      const int sendcounts[],
                                      const int sdispls[],
                                      MPI_Datatype sendtype,
                                      void* recvbuf,
                                      const int recvcounts[],
                                      const int rdispls[],
                                      MPI_Datatype recvtype,
                                      MPIL_Topo* topo,
                                      MPIL_Comm* comm,
                                      MPIL_Info* info,
                                      MPIL_Request** request_ptr);
int MPIL_Neighbor_alltoallv_init_ext_topo(const void* sendbuf,
                                          const int sendcounts[],
                                          const int sdispls[],
                                          const long global_sindices[],
                                          MPI_Datatype sendtype,
                                          void* recvbuf,
                                          const int recvcounts[],
                                          const int rdispls[],
                                          const long global_rindices[],
                                          MPI_Datatype recvtype,
                                          MPIL_Topo* topo,
                                          MPIL_Comm* comm,
                                          MPIL_Info* info,
                                          MPIL_Request** request_ptr);


//Sparse Collumn alltoall alogrithms----------------------------------------------------------------------
enum AlltoallCRSMethod
{
    ALLTOALL_CRS_RMA,
	ALLTOALL_CRS_NONBLOCKING,
	ALLTOALL_CRS_NONBLOCKING_LOC,
	ALLTOALL_CRS_PERSONALIZED,
	ALLTOALL_CRS_PERSONALIZED_LOC
};
extern enum AlltoallCRSMethod mpil_alltoall_crs_implementation;
void MPIL_set_alltoall_crs(enum AlltoallCRSMethod algorithm);
int MPIL_Alltoall_crs(const int send_nnz,
                      const int* dest,
                      const int sendcount,
                      MPI_Datatype sendtype,
                      const void* sendvals,
                      int* recv_nnz,
                      int** src_ptr,
                      int recvcount,
                      MPI_Datatype recvtype,
                      void** recvvals_ptr,
                      MPIL_Info* xinfo,
                      MPIL_Comm* xcomm);

//Sparse Collumn alltoallv algorithms-----------------------------------------------------------------------
enum AlltoallvCRSMethod 
{
	ALLTOALLV_CRS_NONBLOCKING,
	ALLTOALLV_CRS_NONBLOCKING_LOC,
	ALLTOALLV_CRS_PERSONALIZED,
	ALLTOALLV_CRS_PERSONALIZED_LOC
};
extern enum AlltoallvCRSMethod mpil_alltoallv_crs_implementation;
void MPIL_set_alltoallv_crs(enum AlltoallvCRSMethod algorithm);
int MPIL_Alltoallv_crs(const int send_nnz,
                       const int send_size,
                       const int* dest,
                       const int* sendcounts,
                       const int* sdispls,
                       MPI_Datatype sendtype,
                       const void* sendvals,
                       int* recv_nnz,
                       int* recv_size,
                       int** src_ptr,
                       int** recvcounts_ptr,
                       int** rdispls_ptr,
                       MPI_Datatype recvtype,
                       void** recvvals_ptr,
                       MPIL_Info* xinfo,
                       MPIL_Comm* comm);

//Utility functions (used in some of the crs tests, may move internal
int MPIL_Alloc(void** pointer, const int bytes);
int MPIL_Free(void* pointer);

#ifdef __cplusplus
}
#endif

#endif
