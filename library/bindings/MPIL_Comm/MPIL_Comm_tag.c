#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"

int MPIL_Comm_tag(MPIL_Comm* comm, int* tag)
{
    return get_tag(comm, tag);
}