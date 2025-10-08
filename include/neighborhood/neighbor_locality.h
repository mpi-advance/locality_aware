#include <algorithm>
#include <map>
#include <vector>

#include "neighbor.h"
#include "neighbor_init.h"
#include "neighbor_persistent.h"

/******************************************
 ****
 **** Helper Methods
 ****
 ******************************************/

void map_procs_to_nodes(LocalityComm* locality,
                        const int orig_num_msgs,
                        const int* orig_procs,
                        const int* orig_counts,
                        std::vector<int>& msg_nodes,
                        std::vector<int>& msg_node_to_local,
                        bool incr);
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
                     LocalityComm* locality,
                     const int tag);
void form_global_comm(CommData* local_data,
                      CommData* global_data,
                      std::vector<int>& local_data_nodes,
                      const MPIL_Comm* mpix_comm,
                      int tag);
void update_global_comm(LocalityComm* locality);
void form_global_map(const CommData* map_data, std::map<long, int>& global_map);
void map_indices(CommData* idx_data, std::map<long, int>& global_map);
void map_indices(CommData* idx_data, const CommData* map_data);
void remove_duplicates(CommData* comm_pkg);
void remove_duplicates(CommPkg* data);
void remove_duplicates(LocalityComm* locality);
void update_indices(LocalityComm* locality,
                    std::map<long, int>& send_global_to_local,
                    std::map<long, int>& recv_global_to_local);
