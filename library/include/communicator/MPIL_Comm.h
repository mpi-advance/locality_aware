#ifndef MPIL_COMM_H
#define MPIL_COMM_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif
/** @brief Struct capable of maintaining multiple request and communicators for library operations.
 *  @details
 *     Protected struct, external user access must be through MPIL APIs.
 *     Supported operations include:
 *         - Buffers for MPI_Windows
 *         - MPI_comms for locality, multileader, and neighborhoods
 *     Contains preprocessor locked GPU aware components. 
**/
typedef struct _MPIL_Comm
{
	/**@brief Global MPI comm for reference, usually MPI_COMM_WORLD**/
    MPI_Comm global_comm;

	/**@brief communicator containing neighbor processes**/
    MPI_Comm neighbor_comm;

    // For hierarchical collectives
	/**@brief Communicator for communicating inside the node**/
    MPI_Comm local_comm;
	/**@brief Communicator containing leader process on each node**/
    MPI_Comm group_comm; 

	/**@brief Communicator containing a single leader and its subordinates**/
    MPI_Comm leader_comm;
	/**@brief Communicator containing all leaders **/
    MPI_Comm leader_group_comm;
	/**@brief Communicator containing all leaders on a single node**/
    MPI_Comm leader_local_comm;

	/**@brief Number of nodes in comm**/
    int num_nodes;
	/**@brief Rank of process in comm**/
    int rank_node;
	/**@brief Processes per node**/
    int ppn;

	/**@brief MPI_window if using sync**/
    MPI_Win win;
	/**@brief Buffer for MPI_window**/
    char* win_array;
	/**@brief Size of win_array in bytes**/
    int win_bytes;
	/**@brief Size of the datatype in win_array in bytes**/
    int win_type_bytes;

	/**@brief Internal array of requests made during a blocking collective**/
    MPI_Request* requests;
	/**@brief Status the requests in requests**/
    MPI_Status* statuses;
	
	/**@brief Size of requests and statuses
	   @details 
			requests and statuses should always be the same size.
			can be updated through MPIL_Comm_req_resize;
	**/
    int n_requests;
    /** @brief Unique identifier for any requests using this comm (defaulting to 126)**/
    int tag;
	/** @brief Maximum size of tag allowed by the system.**/
    int max_tag;

	/** @brief Maps rank in global_comm to rank in local_comm **/
    int* global_rank_to_local;
	/** @brief Maps rank in global_comm to node id (0 based) **/
    int* global_rank_to_node;
	/** @brief Orders ranks bases on node, node*ppn+local **/
    int* ordered_global_ranks;

#ifdef GPU
	/** @brief Number of gpus on the node**/
    int gpus_per_node;
	/** @brief Rank running on the gpu**/
    int rank_gpu;
    /** @brief Pointer to gpuStream_t
		@details 
			Changed to void* to assist compiling.
			Actual type is gpuStream_t, changed to void* to assist compiling.
	*/
    void* proc_stream;
#endif
} MPIL_Comm;

/** @brief Returns the node that process proc is on(data->global_rank_to_node[proc]**/
int get_node(const MPIL_Comm* data, const int proc);

/** @brief Return the rank of proc in local communicator (using MPIL_Comm::global_rank_to_local)**/
int get_local_proc(const MPIL_Comm* data, const int proc);

/** @brief Given a node and a rank of a process, get its rank in the global communicator**/
int get_global_proc(const MPIL_Comm* data, const int node, const int local_proc);

// For testing purposes (manually set PPN)
/** @brief Recreates internal communicators given the number of processes per node
	@details
		Frees and resets local_com and group_comm.
		Splits MPIL_Comm::global_comm into local_comms of size ppn or smaller.
		Remaps rank to rank in local_comms, each node, and reorders_global
**/
int update_locality(MPIL_Comm* xcomm, int ppn);

/** @brief Gets current tag from xcomm then increments MPIL_Comm::tag 
	@details
	  Invoked externally by MPIL_Comm_get_tag
	@param [in, out] xcomm communicator to query and updated
	@param [out] tag value of xcomm->tag before the operations
	@return MPI_SUCCESS
**/
int get_tag(MPIL_Comm* xcomm, int* tag);

#ifdef __cplusplus
}
#endif

#endif