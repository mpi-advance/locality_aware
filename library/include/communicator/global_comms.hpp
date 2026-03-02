#ifndef LA_GLOBAL_COMM_HPP
#define LA_GLOBAL_COMM_HPP

#include <mpi.h>

namespace Communicator
{
    /**@brief Global MPI Communicator to replace MPI_COMM_WORLD inside this library */
    extern MPI_Comm WORLD_COMM;
}  // namespace Communicator

#endif