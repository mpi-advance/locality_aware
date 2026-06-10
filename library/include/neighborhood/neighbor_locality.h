#ifndef NEIGHBOR_LOCALITY_H
#define NEIGHBOR_LOCALITY_H

#include <map>
#include <vector>

#include "communicator/MPIL_Comm.hpp"
#include "comm_data.h"

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

void map_indices(CommData* idx_data, std::map<long, int>& global_map);
void map_indices(CommData* idx_data, const CommData* map_data);


#endif
