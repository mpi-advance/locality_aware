#ifndef NEIGHBOR_LOCALITY_H
#define NEIGHBOR_LOCALITY_H

#include <map>
#include <vector>

#include "communicator/MPIL_Comm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _CommData
{
    int num_msgs;
    int size_msgs;
    int datatype_size;
    int* procs;
    int* indptr;
    int* counts;
    int* indices;
    char* buffer;
} CommData;

  

void map_procs_to_nodes(const int orig_num_msgs,
                        const int* orig_procs,
                        const int* orig_counts,
                        std::vector<int>& msg_nodes,
                        std::vector<int>& msg_node_to_local,
                        bool incr,
                        MPIL_Comm* mpil_comm);
void form_local_comm(const int orig_num_sends,
                     const int* orig_send_procs,
                     const int* orig_send_ptr,
                     const int* orig_sendcounts,
                     const long* orig_send_indices,
                     const std::vector<int>& nodes_to_local,
                     CommData* send_data,
                     CommData* recv_data,
                     CommData* local_data,
                     std::vector<int>& recv_idx_nodes,
                     MPIL_Comm* mpil_comm);
void form_global_comm(CommData* local_data,
                      CommData* global_data,
                      std::vector<int>& local_data_nodes,
                      MPIL_Comm* mpil_comm);
void update_global_comm(CommData* global_send_data,
                        CommData* global_recv_data,
                        MPIL_Comm* mpil_comm);
void form_global_map(const CommData* map_data, std::map<long, int>& global_map);
void remove_duplicates(CommData* comm_pkg);


void destroy_comm_data(CommData* data);
/** @brief Sets the CommData::num_msgs to provided value */
void init_num_msgs(CommData* data, int num_msgs);
/** @brief Sets CommData::size_msgs and allocates CommData::indices for indexing messages **/
void init_size_msgs(CommData* data, int size_msgs);

#ifdef __cplusplus
}
#endif

#endif
