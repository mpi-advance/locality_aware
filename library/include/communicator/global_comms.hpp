#ifndef LA_GLOBAL_COMM_HPP
#define LA_GLOBAL_COMM_HPP

#include <mpi.h>

#include <stdexcept>

namespace Communicator
{
    /**@brief Global variable for the MPI Communicator of all on-node processes */
    extern MPI_Comm NODE_COMM;
    /**@brief Global MPI Communicator to replace MPI_COMM_WORLD inside this library */
    extern MPI_Comm WORLD_COMM;

    /**@brief Internal setup function for creating libray-specific global communicators
     * @details
     *      Will setup ::WORLD_COMM (as a MPI_COMM_DUP of provided communicator) and
     *      ::NODE_COMM (as a MPI_COMM_SPLIT_TYPE of provided communicator)
     * @param [in] world The MPI communicator to use as the base "world" comm
     */
    static void initialize_communicators(MPI_Comm world)
    {
        if (MPI_COMM_NULL != WORLD_COMM || MPI_COMM_NULL != NODE_COMM)
        {
            throw std::runtime_error("Internal Communicators already initialized");
        }

        MPI_Comm_dup(world, &WORLD_COMM);

        MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &NODE_COMM);
    }

    static void teardown_communicators()
    {
        MPI_Comm_free(&WORLD_COMM);
        MPI_Comm_free(&NODE_COMM);
    }

}  // namespace Communicator

#endif