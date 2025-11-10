#ifndef MPIL_COMM_H
#define MPIL_COMM_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif
/** @brief struct capable of maintaining multiple request and communicators for library operations.
 *  @details
 *     Protected struct, user access must be through init, get, or set style functions. 
 *     Supported operations include
 *        buffers for MPI_Windows
 *        MPI_comm slots for locality, multileader, and neighborhoods
 *     Contains preprocessor locked GPU aware components. 
**/
typedef struct _MPIL_Comm
{
	/**@brief global MPI comm for reference, usually MPI_COMM_WORLD**/
    MPI_Comm global_comm;

	/**@brief communicator containing neighbor processes**/
    MPI_Comm neighbor_comm;

    // For hierarchical collectives
	/**@brief communicator for communicating inside the node**/
    MPI_Comm local_comm;
	/**@brief communicator containing leader process on each node**/
    MPI_Comm group_comm; 

	/**@brief comm containing group leaders**/
    MPI_Comm leader_comm;
	/**@brief comm containing **/
    MPI_Comm leader_group_comm;
	/**@brief comm containing local processes under leader**/
    MPI_Comm leader_local_comm;

	/**@brief number of nodes in comm**/
    int num_nodes;
	/**@brief rank of process in comm**/
    int rank_node;
	/**@brief processes per node**/
    int ppn;

	/**@brief MPI_window if using sync**/
    MPI_Win win;
	/**@brief buffer for MPI_window**/
    char* win_array;
	/**@brief size of win_array in bytes**/
    int win_bytes;
	/**@brief size of the datatype in win_array in bytes**/
    int win_type_bytes;

	/**@brief array of requests using the comm **/
    MPI_Request* requests;
	/**@brief status the requests in this->requests**/
    MPI_Status* statuses;
	/**@brief size of requests and statuses
	   @details 
			requests and statuses should always be the same size.
			can be updated through MPIL_Comm_req_resize;
	**/
    int n_requests;
    /** @brief unique identifier for the comm **/
    int tag;
	/** @brief maximum size of tag allowed by the system (currently hard capped at 126)**/
    int max_tag;

	/** @brief maps rank in global_comm to rank in local_comm **/
    int* global_rank_to_local;
	/** @brief maps rank in global_comm to node id (0 based) **/
    int* global_rank_to_node;
	/** @brief orders ranks bases on node, node*ppn+local **/
    int* ordered_global_ranks;

#ifdef GPU
	/** @brief number of gpus on the node**/
    int gpus_per_node;
	/** @brief rank running on the gpu**/
    int rank_gpu;
    /** @brief pointer to gpuStream_t
		@details changed to void* to assist compiling.
	*/
	// actual type is gpuStream_t, changed to void* to assist compiling.
    void* proc_stream;
#endif
} MPIL_Comm;

/** @brief wrapper around data->global_rank_to_node**/
int get_node(const MPIL_Comm* data, const int proc);

/** @brief wrapper around data->global_rank_to_local**/
int get_local_proc(const MPIL_Comm* data, const int proc);

/** @brief wrapper around data->ordered_global_ranks**/
int get_global_proc(const MPIL_Comm* data, const int node, const int local_proc);

// For testing purposes (manually set PPN)
/** @brief recreates internal comms given the number of processes per node
	@details
		frees and resets local_com and group_comm
		splits global into local_comms of size ppn or smaller
		remaps global rank to rank in local_comms, each node, and reorders_global
**/
int update_locality(MPIL_Comm* xcomm, int ppn);

/** @brief gets current tag from xcomm then increments xcomm->tag 
	@details
	  invoked externally by MPIL_Comm_get_tag
	@param [in, out] xcomm communicator to query and updated
	@param [out] tag value of xcomm->tag before the operations
	@return MPI_SUCCESS
**/
int get_tag(MPIL_Comm* xcomm, int* tag);

#ifdef __cplusplus
}
#endif

#endif