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
 *    @brief Enumerations of implemented algorithms 
 *	@details Each member has
 *	one or more descriptors after the main function
 *	that change the underlying algorithm. 
 *	When supplied to the algorithm selection function. 
 *		<br> STANDARD: Uses standard collective operation
 *		<br>PAIRWISE: Uses Pairwise communication pattern. 
 *		<br>NONBLOCKING: Uses non-blocking communication internally. 
 *		<br>HIERARCHICAL: Single leader aggregates messages before redistribution
 *		<br>MULTILEADER: Group of leaders aggregates message among themselves  before distribution. 
 *		<br>LOCALITY_AWARE|LOCALITY|LOC: Uses hardware architecture based  *communication tree for operations.    
 *		<br>NODE_AWARE: Partial locality awareness (limited to node level locality)
 *		<br>BATCH: Aggregate messages for bulk send
 *		<br>ASYNC: Uses RMA for one-sided communication 
 *		<br>GPU: Offload communication to GPU 
 *		<br>CTC: Copy messages from gpu to cpu before sending. 
 *		<br>INIT: Persistent Communication used internally. 
 *		<br>PMPI: Calls the underlying MPI implementation. 
**/

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

enum AllreduceMethod
{
#if defined(GPU)
#if defined(GPU_AWARE)
    ALLREDUCE_GPU_RECURSIVE_DOUBLING,
    ALLREDUCE_GPU_DISSEMINATION_LOC,
    ALLREDUCE_GPU_DISSEMINATION_ML,
    ALLREDUCE_GPU_DISSEMINATION_RADIX,
    ALLREDUCE_GPU_PMPI,
#endif
    ALLREDUCE_CTC_RECURSIVE_DOUBLING,
    ALLREDUCE_CTC_DISSEMINATION_LOC,
    ALLREDUCE_CTC_DISSEMINATION_ML,
    ALLREDUCE_CTC_DISSEMINATION_RADIX,
    ALLREDUCE_CTC_PMPI,
#endif
    ALLREDUCE_RECURSIVE_DOUBLING,
    ALLREDUCE_DISSEMINATION_LOC,
    ALLREDUCE_DISSEMINATION_ML,
    ALLREDUCE_DISSEMINATION_RADIX,
    ALLREDUCE_PMPI
};

enum AllgatherMethod
{
#if defined(GPU)
#if defined(GPU_AWARE)
    ALLGATHER_GPU_RING,
    ALLGATHER_GPU_BRUCK,
    ALLGATHER_GPU_PMPI,
#endif
    ALLGATHER_CTC_RING,
    ALLGATHER_CTC_BRUCK,
    ALLGATHER_CTC_PMPI,
#endif
    ALLGATHER_RING,
    ALLGATHER_BRUCK,
    ALLGATHER_PMPI
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

/** \defgroup globals Algorithm_switches
 *@brief Global variables used to select which algorithms to use inside MPIL API calls.
 *@{
*/
extern enum AlltoallMethod mpil_alltoall_implementation;
extern enum AlltoallvMethod mpil_alltoallv_implementation;
extern enum AllreduceMethod mpil_allreduce_implementation;
extern enum AllgatherMethod mpil_allgather_implementation;
extern enum NeighborAlltoallvMethod mpil_neighbor_alltoallv_implementation;
extern enum NeighborAlltoallvInitMethod mpil_neighbor_alltoallv_init_implementation;
extern enum AlltoallCRSMethod mpil_alltoall_crs_implementation;
extern enum AlltoallvCRSMethod mpil_alltoallv_crs_implementation;
extern int mpil_collective_radix;
 /**@}*/
 

/** \defgroup global_setters algorithm_set_functions 
 * @brief Functions to set global settings to call a chosen algorithm.
 * @details
 * Each accepts an enumerated value from one of the Algorithm enumerations.
 * If an invalid value is provided, called MPIL functions will use their default algorithm. 
 *
 * @{
*/
int MPIL_Set_alltoall_algorithm(enum AlltoallMethod algorithm);
int MPIL_Set_alltoallv_algorithm(enum AlltoallvMethod algorithm);
int MPIL_Set_allreduce_algorithm(enum AllreduceMethod algorithm);
int MPIL_Set_allgather_algorithm(enum AllgatherMethod algorithm);
int MPIL_Set_alltoallv_neighbor_alogorithm(enum NeighborAlltoallvMethod algorithm);
int MPIL_Set_alltoallv_neighbor_init_alogorithm(
    enum NeighborAlltoallvInitMethod algorithm);
int MPIL_Set_alltoall_crs(enum AlltoallCRSMethod algorithm);
int MPIL_Set_alltoallv_crs(enum AlltoallvCRSMethod algorithm);
int MPIL_Set_collective_radix(int radix);
 /**@}*/

/**@brief Allocate and initialize MPIL_Comm object at xcomm using global_comm as a parent. 
 * @details
 *   Will set xcomm->global_comm to global_comm and 
 *   sets tag value up to 126 based on MPI_Comm_world
 * 
 * @param [in] global_comm parent communicator
 * @param [out] xcomm newly initilizied MPIL_Comm object.  
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_init(MPIL_Comm** xcomm_ptr, MPI_Comm global_comm);

/** @brief Free MPIL_Comm and all contained structs. 
 * @param [in, out] xcomm_ptr pointer to xcomm object to delete.  
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_free(MPIL_Comm** xcomm_ptr);

/** @brief Create comm for communication between processes on the same node. 
 * @details
 *     Sets up LocalityComm by  gathering arrays for get_node, get_local, and get_global methods.
 *     These arrays allow for these methods to work with any ordering.
 *     No longer relying on SMP ordering of processes to nodes!
 *     This functions assumes all nodes have the same number of processes per node. 
 * 
 * @param [in, out] xcomm MPIL_comm to modify
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_topo_init(MPIL_Comm* xcomm);

/** @brief Delete local_comms
 * @details 
 *   Called by MPIL_Comm_free.
 *   Deletes MPIL_topo object stored by xcomm. 
 * @param [in, out] xcomm MPIL_comm to modify
 * @return MPI_Success upon successful completion. 
*/
int MPIL_Comm_topo_free(MPIL_Comm* xcomm);

/** @brief Generates MPIL_Topo from using comm object returned from MPIL_Dist_graph_create_adjacent.
 * @details 
 *    Uses neighbor_comm inside comm to generate MPIL_Topo object. 
 *    Calls MPI_Dist_graph_neighbors_count and MPI_Dist_graph_neighbors. 
 *		
 * @param [in] comm MPIL_comm object with set neighbor_comm 
 * @param [out] mpil_topo_ptr pointer to generated MPIL_Topo struct
 * @return MPI_Success upon successful completion. 
*/
int MPIL_Topo_from_neighbor_comm(MPIL_Comm* comm, MPIL_Topo** mpil_topo_ptr);

/** @brief Subdivides an MPIL_Comm into subgroups and assigns communication leader for each subgroup. 
 * @details
 *     Global communicator is split into sub-communicators of maximum size proc_per_leader.
 *     Calls MPIL_Comm_topo_init if MPIL_Comm::local_comm is not set.      
 *     Leader is selected from each sub-communicator and added to MPIL_Comm::leader_group_comm
 *
 * @param [in, out] xcomm MPIL_comm to divide 
 * @param [in] procs_per_leader maximum number of processes per leader 
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_leader_init(MPIL_Comm* xcomm, int procs_per_leader);

/** @brief Free MPI communicators created as part of hierarchical or multi-leader setups.  
 * @details
 *    Called by MPIL_Comm_free 
 * @param [in, out] MPIL_Comm  
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_leader_free(MPIL_Comm* xcomm);

/** @brief Create a window for one-sided communication. 
 * @details 
 *    Allocate memory and create MPI_Window.    
 * 
 * @param [in, out] xcomm
 * @param [in] bytes size of windows in bytes
 * @param [in] type_bytes local_unit_size for displacements, in bytes
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_win_init(MPIL_Comm* xcomm, int bytes, int type_bytes);

/** @brief Delete window and free allocated memory, called by MPIL_Comm_free**/
int MPIL_Comm_win_free(MPIL_Comm* xcomm);

/** @brief Initialize GPU stream inside the communicator
 * @details
 * GPU Stream is mapped based on selected support. 
 * If at least one gpu is detected, creates gpuStream and binds to xcomm. 
 * If not built with GPU support, this function is a no-op 
 * 
 * @param [in,out] xcomm MPIL_Comm to initialize stream for GPU. 
 * @return MPI_Success upon successful completion. 
.**/

int MPIL_Comm_device_init(MPIL_Comm* xcomm);
/** @brief Destroys the gpustream operated by xcomm
 * @details
 *   If not built with GPU support, this function is a no-op 
 *
 * @param [in,out] xcomm MPIL_Comm object holding handle of gpustream to deallocate. 
 * @return MPI_Success upon successful completion. 
.**/
int MPIL_Comm_device_free(MPIL_Comm* xcomm);

/** @brief Resize xcomm number of requests and status arrays to be size n.**/
int MPIL_Comm_req_resize(MPIL_Comm* xcomm, int n);

/** @brief Wrapper around update_locality, see update_locality().**/
int MPIL_Comm_update_locality(MPIL_Comm* xcomm, int ppn);

/** @brief Get current tag in communicator and then increment tag by 1. See get_comm()**/
int MPIL_Comm_tag(MPIL_Comm* comm, int* tag);

// Functions to initialize and free the MPI_Info object
/** @brief Initializes MPIL_Info object**/
int MPIL_Info_init(MPIL_Info** info);

/** @brief Deletes MPIL_Info object**/
int MPIL_Info_free(MPIL_Info** info);

/** @brief Initializes and returns pointer to ::MPIL_Topo object **/
int MPIL_Topo_init(int indegree,
                   const int sources[],
                   const int sourceweights[],
                   int outdegree,
                   const int destinations[],
                   const int destweights[],
                   MPIL_Info* info,
                   MPIL_Topo** mpil_topo_ptr);

/** @brief deletes ::MPIL_topo object **/
int MPIL_Topo_free(MPIL_Topo** topo);

/**@brief Start processing the request. 
  * @details
  *	 Query request::start_function and call it to activate the request. 
**/
int MPIL_Start(MPIL_Request* request);

/**@brief Wait for the request to complete.  
 *  @details 
 * Query request::wait_function and call it to wait for requests to complete. 
**/
int MPIL_Wait(MPIL_Request* request, MPI_Status* status);

/**@brief Deallocates MPIL_Request object and any internal structures **/
int MPIL_Request_free(MPIL_Request** request);

/** @brief Set reorder value of request to value **/
int MPIL_Request_reorder(MPIL_Request* request, int value);

/** @brief Wrapper around MPI_Dist_graph_create_adjacent. */
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
/** @defgroup collective_func Algorithm APIs
 *  @brief Wrapper functions around algorithms to fit MPI semantics
 *  @details
 *	Use switch statements and global variables to call
 *	internal algorithm to complete the associated 
 *		
 *	Parameters are the same as MPI_Alltoall, just adapted to work 
 *	with objects extended by the library. 
**/

/** @brief Wrapper around MPI_Alltoall.  
 *  @details
 *  Defaults to AllTOALL_PMPI
 *	@ingroup collective_func 
 */
int MPIL_Alltoall(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* comm);
				  		  
/** @brief Wrapper around MPI_Alltoallv.  
 *  @details
 *  Defaults to AllTOALLV_PMPI
 *	@ingroup collective_func 
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

/** @brief Wrapper around MPI_Allreduce.
 *  @details
 *  Defaults to AllREDUCE_PMPI
 *  @ingroup collective_func
 */
int MPIL_Allreduce(const void* sendbuf,
                   void* recvbuf,
                   int count,
                   MPI_Datatype datatype,
                   MPI_Op op,
                   MPIL_Comm* comm);

/** @brief Wrapper around MPI_Allgather.
 *  @details
 *  Defaults to AllGATHER_PMPI
 *  @ingroup collective_func
 */
int MPIL_Allgather(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm);

/** @brief Wrapper around MPI_Allreduce_init.
 *  @details
 *  Defaults to AllREDUCE_PMPI
 *  @ingroup collective_func
 */
int MPIL_Allreduce_init(const void* sendbuf,
                   void* recvbuf,
                   int count,
                   MPI_Datatype datatype,
                   MPI_Op op,
                   MPIL_Comm* comm,
                   MPIL_Info* info,
                   MPIL_Request** req_ptr);

/** @brief Wrapper around MPI_Neighbor_alltoallv  
 *	@ingroup collective_func 
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
							
/** @brief Wrapper around MPI_Neighbor_alltoallv that accepts an already generated topology.  
 *	@ingroup collective_func 
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

/** @brief Wrapper around persistent versions MPI_Neighbor_alltoallv.
 *	@ingroup collective_func 
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
								 
/** @brief Extended version of neighbor alltoallv that allows you to provide global indices. 
 *	@ingroup collective_func 
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
									 
/** @brief Wrapper around persistent version MPI_Neighbor_alltoallv that accepts an already generated topology object. 
 *	@ingroup collective_func 
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

/** @brief Extended version of MPIL_Neighbor_alltoallv_init_topo that allows you to provide global indices.  
 *	@ingroup collective_func 
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

/** @brief Extended version of MPI_Alltoall optimized for compressed row storage layout. 
 *  @details
 *  Sets up dynamic communication tree based on row sparsity in the supplied matrix.  	
 * 
 *  
 *
 *  Defaults to ALLTOALL_CRS_PERSONALIZED;
 *	@ingroup collective_func
 *  
 *  @param [in] send_nnz Number of dynamic sends. 
 *  @param [in] dest Destination of the messages
 *  @param [in] sendcount Send per message count
 *  @param [in] sendtype Datatype being sent
 *  @param [in] sendvals Data to be sent. 
 *  @param [in,out] recv_nnz Number of dynamic recvs
 *  @param [out] src_ptr Destinations of messages to be recieved
 *  @param [out] recvcount Receives per-message count
 *  @param [out] recvtype Datatype being recieved
 *  @param [out] recvvals_ptr Data to receive
 *  @param [in] xinfo
 *  @param [in] xcomm
 *  @return MPI_Success
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
					  
/** @brief  Extended version of MPI_Alltoallv optimized for compressed row storage layout. 
 *  @details
 *  Defaults to ALLTOALLV_CRS_PERSONALIZED; 
 * 
 *	@ingroup collective_func 
 *  @param [in] send_nnz Number of dynamic sends. 
 *  @param [in] dest Destination of the messages
 *  @param [in] sendcount Send per message count
 *  @param [in] sdispls Displacements of sent messages. 
 *  @param [in] sendtype Datatype being sent
 *  @param [in] sendvals Data to be sent. 
 *  @param [in, out] recv_nnz Number of dynamic receives
 *  @param [in, out] size of dynamic receives 
 *  @param [out] src_ptr Destination for received message
 *  @param [out] recvcount Receives per-message count
 *  @param [out] recvtype Datatype being received
 *  @param [out] recvvals_ptr Data to receive
 *  @param [in] xinfo
 *  @param [in] xcomm
 *  @return MPI_Success
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


/** @brief Dynamically allocates enough space for char[bytes] and returns pointer to the new allocation. 
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

#if defined(GPU)
/** @brief Allocates data on the gpu with either cudaMalloc or hipMalloc
 **/
int MPIL_GPU_Alloc(void** pointer, const int bytes);
/** @brief Frees device pointer with cudaFree or hipFree **/
int MPIL_GPU_Free(void* pointer);
#endif

#ifdef __cplusplus
}
#endif

#endif
