#ifndef NAP_COMMUNICATE_HPP
#define NAP_COMMUNICATE_HPP

#include "topo_data.hpp"
#include "nap_comm.h"

#include <mpi.h>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>

/******************************************
 ****
 **** Class Structs
 ****
 ******************************************/

struct NAPCommData{
    void* buf;
    char* global_buffer;
    char* local_L_buffer;
    MPI_Datatype datatype;
    int tag;

    NAPCommData()
    {
        buf = NULL;
        global_buffer = NULL;
        local_L_buffer = NULL;
    }

    ~NAPCommData()
    {
        if (global_buffer) delete[] global_buffer;
        if (local_L_buffer) delete[] local_L_buffer;
    }
};

struct NAPData{
    NAPCommData* send_data;
    NAPCommData* recv_data;
    int tag;
    MPI_Comm mpi_comm;

    NAPData()
    {
        send_data = NULL;
        recv_data = NULL;
    }
};



// Data required for a single step of sends or recvs
struct comm_data{
    int num_msgs;
    int size_msgs;
    int* procs;
    int* indptr;
    int* indices;

    comm_data()
    {
        num_msgs = 0;
        size_msgs = 0;
        procs = NULL;
        indptr = NULL;
        indices = NULL;
    }

    ~comm_data()
    {
        delete[] procs;
        delete[] indptr;
        delete[] indices;
    }

    void init_num_data(int n)
    {
        if (n)
            procs = new int[n];
        indptr = new int[n+1];
        indptr[0] = 0;
    }

    void remove_duplicates()
    {
        int start, end;

        for (int i = 0; i < num_msgs; i++)
        {
            start = indptr[i];
            end = indptr[i+1];
            std::sort(indices+start, indices+end);
        }

        size_msgs = 0;
        start = indptr[0];
        for (int i = 0; i < num_msgs; i++)
        {
            end = indptr[i+1];
            indices[size_msgs++] = indices[start];
            for (int j  = start; j < end - 1; j++)
            {
                if (indices[j+1] != indices[j])
                {
                    indices[size_msgs++] = indices[j+1];
                }
            }
            start = end;
            indptr[i+1] = size_msgs;
        }
    }
};

// Data required for a single communication step
struct comm_pkg{
    comm_data* send_data;
    comm_data* recv_data;

    comm_pkg()
    {
        send_data = new comm_data();
        recv_data = new comm_data();
    }

    ~comm_pkg()
    {
        delete send_data;
        delete recv_data;
    }
};

// Data required for an instance of node-aware communication
struct NAPComm{
    comm_pkg* local_L_comm;
    comm_pkg* local_R_comm;
    comm_pkg* local_S_comm;
    comm_pkg* global_comm;
    int buffer_size; // used for buffer
    MPI_Request* send_requests;
    MPI_Request* recv_requests;
    topo_data* topo_info;
    NAPData* nap_data;

    NAPComm(topo_data* _topo_info)
    {
        buffer_size = 0;
        send_requests = NULL;
        recv_requests = NULL;
        local_L_comm = new comm_pkg();
        local_R_comm = new comm_pkg();
        local_S_comm = new comm_pkg();
        global_comm = new comm_pkg();
        topo_info = _topo_info;
        nap_data = new NAPData();
    }

    ~NAPComm()
    {
        delete local_L_comm;
        delete local_R_comm;
        delete local_S_comm;
        delete global_comm;

        delete[] send_requests;
        delete[] recv_requests;

        delete nap_data;
    }

    void finalize()
    {
        int tmp, max_n;

        // Find max size sent (for send_buffer)
        buffer_size = local_L_comm->send_data->size_msgs;
        tmp = local_S_comm->send_data->size_msgs;
        if (tmp > buffer_size) buffer_size = tmp;
        tmp = local_R_comm->send_data->size_msgs;
        if (tmp > buffer_size) buffer_size = tmp;
        tmp = global_comm->send_data->size_msgs;
        if (tmp > buffer_size) buffer_size = tmp;

        // Find max number sent and recd
        max_n = local_L_comm->send_data->num_msgs + 
            global_comm->send_data->num_msgs;
        tmp = local_S_comm->send_data->num_msgs;
        if (tmp > max_n) max_n = tmp;
        tmp = local_R_comm->send_data->num_msgs;
        if (tmp > max_n) max_n = tmp;
        if (max_n) send_requests = new MPI_Request[max_n];

        // Find max number sent and recd
        max_n = local_L_comm->recv_data->num_msgs + 
            global_comm->recv_data->num_msgs;
        tmp = local_S_comm->recv_data->num_msgs;
        if (tmp > max_n) max_n = tmp;
        tmp = local_R_comm->recv_data->num_msgs;
        if (tmp > max_n) max_n = tmp;
        if (max_n) recv_requests = new MPI_Request[max_n];
    }
};

/******************************************
 ****
 **** Helper Methods
 ****
 ******************************************/
static void map_procs_to_nodes(NAPComm* nap_comm, const int orig_num_msgs,
    const int* orig_procs, const int* orig_indptr,
    std::vector<int>& msg_nodes, std::vector<int>& msg_node_to_local,
    MPI_Comm mpi_comm, bool incr = true);
static void form_local_comm(const int orig_num_sends, const int* orig_send_procs,
    const int* orig_send_ptr, const int* orig_send_indices,
    const std::vector<int>& nodes_to_local, comm_data* send_data,
    comm_data* recv_data, comm_data* local_data,
    std::vector<int>& recv_idx_nodes, MPI_Comm mpi_comm,
    topo_data* topo_info, const int tag);
static void form_global_comm(comm_data* local_data, comm_data* global_data,
    std::vector<int>& local_data_nodes, MPI_Comm mpi_comm,
    topo_data* topo_info, int tag);
static void update_global_comm(NAPComm* nap_comm, topo_data* topo_info, MPI_Comm mpi_comm);
static void update_indices(NAPComm* nap_comm, std::map<int, int>& send_global_to_local,
        std::map<int, int>& recv_global_to_local);


static void MPIX_step_comm(comm_pkg* comm, const void* send_data, char** recv_data,
        int tag, MPI_Comm local_comm, MPI_Datatype send_type,
        MPI_Datatype recv_type, MPI_Request* send_requests,
        MPI_Request* recv_requests);
static void MPIX_step_send(comm_pkg* comm, const void* send_data,
        int tag, MPI_Comm mpi_comm, MPI_Datatype datatype,
        MPI_Request* send_request, char** send_buffer_ptr);
static void MPIX_step_recv(comm_pkg* comm,
        int tag, MPI_Comm mpi_comm, MPI_Datatype datatype,
        MPI_Request* recv_request, char** recv_buffer_ptr);
static void MPIX_step_waitall(comm_pkg* comm, MPI_Request* send_requests,
        MPI_Request* recv_requests);
static void MPIX_intra_recv_map(comm_pkg* comm, char* intra_recv_data,
        void* inter_recv_data, MPI_Datatype datatype, MPI_Comm mpi_comm);
char* MPIX_NAP_unpack(char* packed_buf, int size, MPI_Datatype datatype, MPI_Comm comm);

#endif
