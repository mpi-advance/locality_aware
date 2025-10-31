#ifndef MPIL_INFO_H
#define MPIL_INFO_H

#ifdef __cplusplus
extern "C" {
#endif

/** @brief struct for containing number of initized columna and overall size

 \todo why a new struct instead of adding key to existing MPI Info object?
**/
typedef struct _MPIL_Info
{
    int crs_num_initialized;
    int crs_size_initialized;
} MPIL_Info;

#ifdef __cplusplus
}
#endif

#endif