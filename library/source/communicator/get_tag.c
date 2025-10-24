#include "communicator/MPIL_Comm.h"

int get_tag(MPIL_Comm* xcomm, int* tag)
{
    *tag       = xcomm->tag;
    xcomm->tag = ((xcomm->tag + 1) % xcomm->max_tag);

    return MPI_SUCCESS;
}
