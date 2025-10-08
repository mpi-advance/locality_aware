#include "../../../../include/communicator/mpil_comm.h"



int MPIL_Comm_tag(MPIL_Comm* xcomm, int* tag)
{
    *tag       = xcomm->tag;
    xcomm->tag = ((xcomm->tag + 1) % xcomm->max_tag);

    return MPI_SUCCESS;
}
