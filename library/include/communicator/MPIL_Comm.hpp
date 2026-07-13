#ifndef MPIL_COMM_H
#define MPIL_COMM_H

#include <algorithm>

#include "global_comms.hpp"

/** @brief Struct capable of maintaining multiple request and communicators for library
 * operations.
 *  @details
 *     Protected struct, external user access must be through MPIL APIs.
 *     Supported operations include:
 *         - Buffers for MPI_Windows
 *         - MPI_comms for locality, multileader, and neighborhoods
 *     Contains preprocessor locked GPU aware components.
 **/
typedef struct _MPIL_Comm
{
    /** @brief Global MPI comm for reference, usually MPI_COMM_WORLD **/
    MPI_Comm global_comm;

    /** @brief communicator containing neighbor processes **/
    MPI_Comm neighbor_comm;

    // For hierarchical collectives
    /** @brief Reference-counted MPI_Comm for communicating inside the node **/
    Communicator::CachedComm local_comm;
    /** @brief Reference-counted MPI_Comm containing leader process on each node **/
    Communicator::CachedComm group_comm;

    /** @brief Communicator containing a single leader and its subordinates **/
    Communicator::CachedComm leader_comm;
    /** @brief Communicator containing all leaders **/
    Communicator::CachedComm leader_group_comm;
    /** @brief Communicator containing all leaders on a single node **/
    Communicator::CachedComm leader_local_comm;

    /** @brief Number of nodes in comm **/
    int num_nodes;
    /** @brief Rank of process in comm **/
    int rank_node;
    /** @brief Processes per node **/
    int ppn;

    /** @brief MPI_window if using sync **/
    MPI_Win win;
    /** @brief Buffer for MPI_window **/
    char* win_array;
    /** @brief Size of win_array in bytes **/
    int win_bytes;
    /** @brief Size of the datatype in win_array in bytes **/
    int win_type_bytes;

    /** @brief Internal array of requests made during a blocking collective **/
    MPI_Request* requests;
    /** @brief Status the requests in requests **/
    MPI_Status* statuses;

    /** @brief Size of requests and statuses
     *  @details requests and statuses should always be the same size. can be updated
     * through MPIL_Comm_req_resize;
     **/
    int n_requests;
    /** @brief Unique identifier for any requests using this comm (defaulting to 126) **/
    int tag;
    /** @brief Maximum size of tag allowed by the system. **/
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
     * @details Changed to void* to assist compiling. Actual type is gpuStream_t, changed
     * to void* to assist compiling.
     **/
    void* proc_stream;
#endif
} MPIL_Comm;

/** @brief Returns the node that process proc is on (data->global_rank_to_node[proc]) **/
int get_node(const MPIL_Comm* data, const int proc);

/** @brief Return the rank of proc in local communicator (using
 * MPIL_Comm::global_rank_to_local)
 **/
int get_local_proc(const MPIL_Comm* data, const int proc);

/** @brief Given a node and a rank of a process, get its rank in the global
 * communicator
 **/
int get_global_proc(const MPIL_Comm* data, const int node, const int local_proc);

/** @brief Constructor for an ::_MPIL_Comm object.
 * @details Allocates (using malloc) an ::_MPIL_Comm object to be saved into the provided
 * location. The passed MPI Communicator is save into _MPIL_Comm::global_comm ; all other
 * variables are set to NULL, 0, or appropriate null MPI object.
 * @param [in, out] xcomm The location to create the MPIL_Comm at.
 * @param [in] global_comm MPI communicator to use a global communicator in MPIL_Comm
 * @return MPI_SUCCESS
 **/
int initialize_comm_object(MPIL_Comm** xcomm, MPI_Comm global_comm);

/** @brief Initialize the per-node, and group communicators inside an ::_MPIL_Comm.
 * @details Each topology communicator is created via an MPI_Comm_split. The per-node
 * communicator will be created with MPI_Comm_split_type using "MPI_COMM_TYPE_SHARED"
 * unless a "ppn_override" value. In that case, the "node" of each rank will be calculated
 * and used as the color for the MPI_Comm_split. The calculation of the the "node" is
 * determined by the templated parameter. If the template is false, "rank/ppn_override" is
 * used; if the template is true "rank % ppn_override" is used.
 *
 * If the pairing of the MPIL_Comm::global_comm and the provided ppn_override have been
 * used before, this method will bypass the calls to create a new MPI Communicator and
 * will instead pull out the appropriate communicator from
 * Communicator::cached_local_comms to fill MPIL_Comm::local_comm and
 * Communicator::cached_group_comms to fill MPIL_Comm::group_comm.
 *
 * To determine if a global_comm has been used before, the MPI_Group of the MPI_Comm is
 * used. MPI_Groups are used instead of MPI_Comm since 1) duping communicators is a
 * collective operation and 2) users could free the MPI_Comm used inside the
 * ::MPIL_Comm::global_comm, which would result in potential segfaults on future
 * MPI_Comm_compare calls. Since MPI_Group creation is local, and we do not care about the
 * "context" of an MPI_Comm, we can get, store, and compare groups instead. Currently, the
 * comparison is done with the help of std::find_if, and only checks for MPI_IDENT that
 * comes from MPI_Group_compare.
 *
 * This method will free any MPI_Group that does not end up cached; cached MPI_Group
 * objects will be freed via Communicator::clear_comm_caches in MPIL_Finalize.
 *
 * @tparam NUMA Controls how the grouping is made in the case that a PPN override is used.
 * @param [in, out] xcomm The ::_MPIL_Comm to store the topology communicators into.
 * @param [in] ppn_override Optional integer to determine how many processes are node.
 * If set, overrides default creation of MPIL_Comm::local_comm.
 * @return MPI_SUCCESS
 * @sa Communicator::CachedComm
 **/
template <bool NUMA = false>
int initialize_topo_communicator(MPIL_Comm* xcomm, int ppn_override = 0)
{
    int rank;
    MPI_Comm_rank(xcomm->global_comm, &rank);

    // Get the group, since we will compare those
    MPI_Group global_group;
    MPI_Comm_group(xcomm->global_comm, &global_group);

    /* Lambda for searching for if a particular group/ppn combo has been used before. */
    auto search_function = [global_group,
                            ppn_override](const Communicator::MapPairType& mpt) {
        if (std::get<1>(mpt.first) != ppn_override)
        {
            return false;
        }

        int result;
        MPI_Group_compare(global_group, std::get<0>(mpt.first), &result);
        /* Currently only care about MPI_IDENT */
        return (result == MPI_IDENT);
    };

    auto local_comm_iter = std::find_if(Communicator::cached_local_comms.begin(),
                                        Communicator::cached_local_comms.end(),
                                        search_function);

    if (Communicator::cached_local_comms.end() != local_comm_iter)
    { /* Used cached entry, so we can free the MPI_Group */
        MPI_Group_free(&global_group);
        xcomm->local_comm = local_comm_iter->second;
    }
    else
    {
        if (ppn_override > 0)
        { /* Split communicator on a custom number of PPN */
            int color = (NUMA) ? rank % ppn_override : rank / ppn_override;
            MPI_Comm_split(xcomm->global_comm, color, rank, xcomm->local_comm);
        }
        else
        { /* Split global comm into local (per node) communicators */
            MPI_Comm_split_type(xcomm->global_comm,
                                MPI_COMM_TYPE_SHARED,
                                rank,
                                MPI_INFO_NULL,
                                xcomm->local_comm);
        }
        /* Cache new local communicator into map for reuse */
        Communicator::cached_local_comms.push_back(
            {{global_group, ppn_override}, xcomm->local_comm});
    }

    /* Get the group again, since it was either freed above, or cached (which will be
     * freed at end of program) */
    MPI_Comm_group(xcomm->global_comm, &global_group);
    auto group_comm_iter = std ::find_if(Communicator::cached_group_comms.begin(),
                                         Communicator::cached_group_comms.end(),
                                         search_function);

    if (Communicator::cached_group_comms.end() != group_comm_iter)
    { /* Used cached entry, so we can free the MPI_Group */
        MPI_Group_free(&global_group);
        xcomm->group_comm = group_comm_iter->second;
    }
    else
    {
        int local_rank;
        MPI_Comm_rank(xcomm->local_comm, &local_rank);
        MPI_Comm_split(xcomm->global_comm, local_rank, rank, xcomm->group_comm);
        /* Cache new group (per local rank) communicator into map for reuse */
        Communicator::cached_group_comms.push_back(
            {{global_group, ppn_override}, xcomm->group_comm});
    }

    return MPI_SUCCESS;
}

/** @brief Allocate and fill in various process mapping array inside an :_MPIL_Comm object
 * @details This method requires that ::initialize_topo_communicator has been called
 * first. Inside this method, MPIL_Comm::global_rank_to_local,
 * MPIL_Comm::global_rank_to_node, and MPIL_Comm::ordered_global_ranks will (usually) be
 * allocated. Once allocated, the first two will be collected from all ranks using an
 * MPI_Allgather to get complete process mappings. The last array will uses these two to
 * build an inverse mapping. Finally, MPIL_Comm::num_nodes and MPIL_Comm::rank_node will
 * be set.
 * @param [in, out] xcomm The _MPIL_Comm object to fill in.
 * @return MPI_SUCCESS
 **/
int initialize_rank_mapping(MPIL_Comm* xcomm);

/** @brief Free the arrays associated with process mapping inside an :_MPIL_Comm object
 * @details More specifically, this method will free MPIL_Comm::global_rank_to_local,
 * MPIL_Comm::global_rank_to_node, and MPIL_Comm::ordered_global_ranks.
 * @return MPI_SUCCESS
 **/
int free_rank_mapping(MPIL_Comm* xcomm);

/** @brief Gets current tag from xcomm then increments MPIL_Comm::tag
 * @details Invoked externally by MPIL_Comm_get_tag
 * @param [in, out] xcomm communicator to query and updated
 * @param [out] tag value of xcomm->tag before the operations
 * @return MPI_SUCCESS
 **/
int get_tag(MPIL_Comm* xcomm, int* tag);

#endif
