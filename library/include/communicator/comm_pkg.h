#ifndef MPI_ADVANCE_COMM_PKG_H
#define MPI_ADVANCE_COMM_PKG_H

#include "comm_data.h"

/** @brief struct containing buffers meta data for two sides of a communication and tag **/
typedef struct _CommPkg
{
    CommData* send_data;
    CommData* recv_data;
    int tag;
} CommPkg;


void init_comm_pkg(CommPkg** comm_ptr,
                   MPI_Datatype sendtype,
                   MPI_Datatype recvtype,
                   int _tag);
/** @brief calls finalize_comm_data on both internal buffers **/
void finalize_comm_pkg(CommPkg* comm);

/** @brief calls destory_comm_data on both internal buffers **/
void destroy_comm_pkg(CommPkg* comm);

#endif
