#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"

int MPIL_Comm_init(MPIL_Comm** xcomm_ptr, MPI_Comm global_comm)
{
    return initialize_comm_object(xcomm_ptr, global_comm);
}
