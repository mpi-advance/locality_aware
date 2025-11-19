#include "communicator/MPIL_Comm.h"
#include "locality_aware.h"

int MPIL_Comm_update_locality(MPIL_Comm* xcomm, int ppn)
{
    return update_locality(xcomm, ppn);
}