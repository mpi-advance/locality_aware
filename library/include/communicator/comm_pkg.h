#ifndef MPI_ADVANCE_COMM_PKG_H
#define MPI_ADVANCE_COMM_PKG_H

#include "comm_data.h"

/** @brief struct for storing message data
 *	@details  
 *    One comm_pkg per process. One comm_data object for each process send to and received from by that process.  	  
**/
typedef struct _CommPkg
{
	/** @brief information on outgoing messages **/
    CommData* send_data;
	/** @brief information on incoming messages **/
    CommData* recv_data;
	
	/** @brief unique id for the struct. **/
    int tag;
} CommPkg;

/** @brief allocate and initalize comm_pkg **/
void init_comm_pkg(CommPkg** comm_ptr,
                   MPI_Datatype sendtype,
                   MPI_Datatype recvtype,
                   int _tag);
/** @brief calls finalize_comm_data on both internal buffers **/
void finalize_comm_pkg(CommPkg* comm);

/** @brief calls destory_comm_data on both internal buffers **/
void destroy_comm_pkg(CommPkg* comm);

#endif
