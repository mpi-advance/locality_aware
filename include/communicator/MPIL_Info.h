#ifndef MPIL_INFO_H
#define MPIL_INFO_H

#include "mpi.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// MPIL Info Object
/** @brief MPIL_Info object 
	\todo why this instead of just an entry in a normal MPI_Info object?
**/
typedef struct _MPIL_Info
{
    int crs_num_initialized;
    int crs_size_initialized;
} MPIL_Info;

/** @brief Constructor of MPIL_Info object, initialized values = 0**/
int MPIL_Info_init(MPIL_Info** info);

/** @brief deallocate and delete supplied info object **/
int MPIL_Info_free(MPIL_Info** info);

#ifdef __cplusplus
}
#endif

#endif
