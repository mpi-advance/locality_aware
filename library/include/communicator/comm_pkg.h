#ifndef MPI_ADVANCE_COMM_PKG_H
#define MPI_ADVANCE_COMM_PKG_H

#include "comm_data.h"

/** @brief Struct for storing directional message data for a group of processes.
 *	@details
 *    One ::CommPkg per process.
 *    One ::CommData object for each process send to and received from by that process.
 **/
typedef struct _CommPkg
{
    /** @brief Information on outgoing messages **/
    CommData* send_data;
    /** @brief Information on incoming messages **/
    CommData* recv_data;
    /** @brief Tag value to use for communications (see ::get_tag()). **/
    int tag;
} CommPkg;

/** @brief Allocate and initialize a ::CommPkg **/
void init_comm_pkg(CommPkg** comm_ptr,
                   MPI_Datatype sendtype,
                   MPI_Datatype recvtype,
                   int _tag);
/** @brief Calls finalize_comm_data() on CommPkg::send_data and CommPkg::recv_data **/
void finalize_comm_pkg(CommPkg* comm);
/** @brief ::CommPkg destructor that cleans up CommPkg::send_data and CommPkg::recv_data
 * **/
void destroy_comm_pkg(CommPkg* comm);

#endif
