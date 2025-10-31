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

/** \defgroup alg_enum Algorithm enumerations
    @brief Enumerations of implemented algorithms 
	@details Each member has
	one or more descriptors after the main function
	that change the underlying algorithm. 
	When supplied to the algorithm selection function. 
		<br> STANDARD: ????
		<br>PAIRWISE: Uses Pairwise communication pattern. 
		<br>NONBLOCKING: Uses non-blocking communication internally. 
		<br>HIERARCHICAL: Local groups???
		<br>MULTILEADER: Separates world into subgroups ???
		<br>LOCALITY_AWARE|LOCALITY|LOC: Locality_aware  
		<br>NODE_AWARE: Partial locality awareness (limited to node level locality)
		<br>BATCH:
		<br>ASYNC:
		<br>GPU: GPU aware
		<br>CTC: Copy to Cpu
		<br>INIT: Persistent Communication used internally. 
		<br>PMPI: Calls the underlying MPI implementation. 
**/

/* Enums for listing of implemented algorithms */
/** @brief Enumeration of implemented alltoall algorithms @ingroup alg_enum**/
enum AlltoallMethod
{
#if defined(GPU) && defined(GPU_AWARE)
    ALLTOALL_GPU_PAIRWISE,
    ALLTOALL_GPU_NONBLOCKING,
    ALLTOALL_CTC_PAIRWISE,
    ALLTOALL_CTC_NONBLOCKING,
#endif
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

/** @brief Enumeration of implemented alltoallv algorithms @ingroup alg_enum**/
enum AlltoallvMethod
{
#if defined(GPU) && defined(GPU_AWARE)
    ALLTOALLV_GPU_PAIRWISE,
    ALLTOALLV_GPU_NONBLOCKING,
    ALLTOALLV_CTC_PAIRWISE,
    ALLTOALLV_CTC_NONBLOCKING,
#endif
    ALLTOALLV_PAIRWISE,
    ALLTOALLV_NONBLOCKING,
    ALLTOALLV_BATCH,
    ALLTOALLV_BATCH_ASYNC,
    ALLTOALLV_PMPI
};

/** @brief Enumeration of implemented neighborhood alltoall algorithms @ingroup alg_enum**/
enum NeighborAlltoallvMethod
{
    NEIGHBOR_ALLTOALLV_STANDARD,
    NEIGHBOR_ALLTOALLV_LOCALITY
};

/** @brief Enumeration of implemented neighborhood alltoallv algorithms @ingroup alg_enum**/
enum NeighborAlltoallvInitMethod
{
    NEIGHBOR_ALLTOALLV_INIT_STANDARD,
    NEIGHBOR_ALLTOALLV_INIT_LOCALITY
};

/** @brief Enumeration of implemented alltoall compressed row storage algorithms @ingroup alg_enum**/
enum AlltoallCRSMethod
{
    ALLTOALL_CRS_RMA,
    ALLTOALL_CRS_NONBLOCKING,
    ALLTOALL_CRS_NONBLOCKING_LOC,
    ALLTOALL_CRS_PERSONALIZED,
    ALLTOALL_CRS_PERSONALIZED_LOC
};

/** @brief Enumeration of implemented alltoallv compressed row storage algorithms @ingroup alg_enum**/
enum AlltoallvCRSMethod 
{
    ALLTOALLV_CRS_NONBLOCKING,
    ALLTOALLV_CRS_NONBLOCKING_LOC,
    ALLTOALLV_CRS_PERSONALIZED,
    ALLTOALLV_CRS_PERSONALIZED_LOC
};

/* Create global variables for algorithm selection. */
/** \defgroup globals Algorithm_switches
 *@brief global variables used to select which algorithms to use 
 *	inside MPIL API calls.
 *@{
*/
extern enum AlltoallMethod mpil_alltoall_implementation;
extern enum AlltoallvMethod mpil_alltoallv_implementation;
extern enum NeighborAlltoallvMethod mpil_neighbor_alltoallv_implementation;
extern enum NeighborAlltoallvInitMethod mpil_neighbor_alltoallv_init_implementation;
extern enum AlltoallCRSMethod mpil_alltoall_crs_implementation;
extern enum AlltoallvCRSMethod mpil_alltoallv_crs_implementation;
 /**@}*/
 
/* Algorithm selection functions. */
/** \defgroup global_setters algorithm_set_functions 
 * @brief Functions to set global settings to call a chosen algorithm.
 *        Each accepts an enumerated value from one of the Algorithm enumerations.
 *        If an invalid value is provided, called MPIL functions will use their default algorithm. 
 * @{
*/
int MPIL_Set_alltoall_algorithm(enum AlltoallMethod algorithm);
int MPIL_Set_alltoallv_algorithm(enum AlltoallvMethod algorithm);
int MPIL_Set_alltoallv_neighbor_alogorithm(enum NeighborAlltoallvMethod algorithm);
int MPIL_Set_alltoallv_neighbor_init_alogorithm(
    enum NeighborAlltoallvInitMethod algorithm);
int MPIL_Set_alltoall_crs(enum AlltoallCRSMethod algorithm);
int MPIL_Set_alltoallv_crs(enum AlltoallvCRSMethod algorithm);
 /**@}*/

// Functions to control various versions of the MPIL_Comm object---------------------
/** @brief allocate and initialize MPIL_Comm object at xcomm using global_comm as a parent. 
 * @details
 *   sets xcomm->global_comm to global_comm
 *   sets tag value up to 126 based on MPI_Comm_world
 * 
 * @param [in] global_comm parent communicator
 * @param [out] xcomm newly initilizied MPIL_Comm object.  
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_init(MPIL_Comm** xcomm_ptr, MPI_Comm global_comm);

/** @brief free MPIL_Comm and all contained structs. 
 * @param [in, out] xcomm_ptr pointer to xcomm object to delete.  
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_free(MPIL_Comm** xcomm_ptr);

/** @brief Create comm for communication between processes with a shared memory pool. 
 * @details
 *     sets up local_comm
 *     Gather arrays for get_node, get_local, and get_global methods <br>
 *     These arrays allow for these methods to work with any ordering <br>
 *     No longer relying on SMP ordering of processes to nodes! <br>
 *     Does rely on constant ppn <br>
 * 
 * @param [in, out] xcomm MPIL_comm to modify
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_topo_init(MPIL_Comm* xcomm);
/** @brief delete local_comms
 * @details called by MPIL_Comm_free
 *
 * @param [in, out] xcomm MPIL_comm to modify
 * @return MPI_Success upon successful completion. 
*/
int MPIL_Comm_topo_free(MPIL_Comm* xcomm);

/** @brief uses MPI neighbor functions to setup MPIL_Topo
 * @details called by MPIL_Comm_free
 *    Uses neighbor_comm inside xcomm to generate MPIL_Topo object. 
 *    calls MPI_Dist_graph_neighbors_count and MPI_Dist_graph_neighbors. 
 *		
 *    MPI_Dist_graph_neighbors
 * @param [in] xcomm MPIL_comm object with set neighbor_comm 
 * @param [out] mpil_topo_ptr pointer to generated MPIL_Topo struct
 * @return MPI_Success upon successful completion. 
*/
int MPIL_Topo_from_neighbor_comm(MPIL_Comm* comm, MPIL_Topo** mpil_topo_ptr);

/** @brief Create disjoint subcomms of maximum size procs_per_leader from xcomm->global_comm
 * @details
 *     splits global comm into subcoms of maximum size proc_per_leader
 *     Leader is selected from each subcomms, leaders communicate vai leader_group_comm
 *     calls MPIL_Comm_topo_init if local_comm is not set. 
 *     Leaders gather from leader_local_comm
 * 
 * @param [in, out] xcomm MPIL_comm to divide 
 * @param [in, out] procs_per_leader maximum number of processes per leader 
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_leader_init(MPIL_Comm* xcomm, int procs_per_leader);
/** @brief free MPI communicators created as part of hierarchical or multi-leader setups.  
 * @details
 *    Called by MPIL_Comm_free 
 * @param [in, out] MPIL_Comm  
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_leader_free(MPIL_Comm* xcomm);

/** @brief create a window for one-sided communication. 
 * @details 
 *    allocate memory and create MPI_Window.    
 * 
 * @param [in, out] xcomm
 * @param [in] bytes size of windows in bytes
 * @param [in] type_bytes local_unit_size for displacements, in bytes
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_win_init(MPIL_Comm* xcomm, int bytes, int type_bytes);
/** @brief delete window and free allocated memory, called by MPIL_Comm_free**/
int MPIL_Comm_win_free(MPIL_Comm* xcomm);

/** @brief initialize gpustream and bind to comm
 * @details
 * Initializes xcomm using MPIL_Comm_topo_init if null. 
 * If at least one gpu is detected, creates gpuStream and binds to xcomm. 
 * Null process if GPU awareness is not active.
 * 
 * @param [in,out] xcomm MPIL_Comm to manage stream. 
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_device_init(MPIL_Comm* xcomm);
/** @brief destroys the gpu_stream operated by xcomm
 * @details
 *   Null process if GPU awareness is not active. 
 *
 * @param [in,out] xcomm MPIL_Comm to link to stream. 
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_device_free(MPIL_Comm* xcomm);

/** @brief resize xcomm number of requests and status arrays to be size n.**/
int MPIL_Comm_req_resize(MPIL_Comm* xcomm, int n);

/** @brief wrapper around update_locality, see update_locality.**/
int MPIL_Comm_update_locality(MPIL_Comm* xcomm, int ppn);

/** @brief wrapper around get_comm, returns tag and increments comm->tag by 1,see @get_comm.**/
int MPIL_Comm_tag(MPIL_Comm* comm, int* tag);

// Functions to initialize and free the MPI_Info object
int MPIL_Info_init(MPIL_Info** info);
int MPIL_Info_free(MPIL_Info** info);

// Functions to control the MPIL_Topo object
int MPIL_Topo_init(int indegree,
                   const int sources[],
                   const int sourceweights[],
                   int outdegree,
                   const int destinations[],
                   const int destweights[],
                   MPIL_Info* info,
                   MPIL_Topo** mpil_topo_ptr);
int MPIL_Topo_free(MPIL_Topo** topo);

// Functions to control the MPIL_Request object
/**@brief wrapper that calls MPI_start or neighbor start **/
int MPIL_Start(MPIL_Request* request);
/**@brief wrapper that calls MPI_wait or neighbor wait **/
int MPIL_Wait(MPIL_Request* request, MPI_Status* status);
int MPIL_Request_free(MPIL_Request** request);
int MPIL_Request_reorder(MPIL_Request* request, int value);

/** @brief wrapper around MPI_Dist_graph_create_adjacent. 
 *	@ingroup api 
 */
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

// Main MPIL Functions
/** @degroup api Core MPIL Function handles
	@details
		Use switch statements and global variables to call
		internal algorithm to complete the associated 
		
		Parameters are the same as MPI_Alltoall, just adapted to work 
		with objects extended by the library. 
**/
/** @brief wrapper around MPI_alltoall.  
 *  @details
 *  Defaults to AllTOALL_PMPI
 *	@ingroup api 
 */
int MPIL_Alltoall(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* comm);
				  		  
/** @brief wrapper around MPI_alltoallv.  
 *  @details
 *  Defaults to AllTOALLV_PMPI
 *	@ingroup api 
 */		  
int MPIL_Alltoallv(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm);

/** @brief wrapper around MPI_Neighbor_alltoallv  
 *	@ingroup api 
 */
int MPIL_Neighbor_alltoallv(const void* sendbuf,
                            const int sendcounts[],
                            const int sdispls[],
                            MPI_Datatype sendtype,
                            void* recvbuf,
                            const int recvcounts[],
                            const int rdispls[],
                            MPI_Datatype recvtype,
                            MPIL_Comm* comm);
							
/** @brief wrapper around MPI_Neighbor_alltoallv that accepts an all ready generated topology.  
 *	@ingroup api 
 */
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

/** @brief wrapper around persistent versions MPI_Neighbor_alltoallv.
 *	@ingroup api 
 */
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
								 
/** @brief wrapper around persistent version MPI_Neighbor_alltoallv, accepts array of requests rather than single request for operation. 
 *	@ingroup api 
 */
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
									 
/** @brief wrapper around persistent version MPI_Neighbor_alltoallv that accepts an already generated topology object. 
 *	@ingroup api 
 */								 
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

/** @brief wrapper around persistent version MPI_Neighbor_alltoallv, accepts array of requests rather than single request for operation, and a already generated topology struct.  
 *  @details
 *  Defaults to ALLTOALLV_CRS_PERSONALIZED; 
 * 
 *	@ingroup api 
 */
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

/** @brief wrapper around MPI_Alltoall implementations optimized for compressed row storage matrix operations. 
 *  @details
 *  Defaults to ALLTOALL_CRS_PERSONALIZED; 
 * 
 *	@ingroup api 
 */
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
					  
/** @brief wrapper around MPI_Alltoallv implementations optimized for compressed row storage matrix operations. 
 *  @details
 *  Defaults to ALLTOALLV_CRS_PERSONALIZED; 
 * 
 *	@ingroup api 
 */					  
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

// Utility functions (used in some of the crs tests, may move internal
/** @brief Dynamically allocates enough space for char[bytes] and returns pointer to the new alloc. 
  * @details
  * Will cause error if bytes < 0
  * Will return null pointer if bytes == 0
  * @param [out] pointer pointer to allocated space. 
  * @param [in] bytes number of chars to make space 
  * @return MPI_Success on a successful return. 
 */
int MPIL_Alloc(void** pointer, const int bytes);
/** @brief Frees space at pointer. 
  * @details
  *    Does nothing if supplied with nullptr. 
  * @param [in, out] pointer to allocated space. 
  * @return MPI_Success on a successful return. 
 */
int MPIL_Free(void* pointer);

#ifdef __cplusplus
}
#endif

#endif
