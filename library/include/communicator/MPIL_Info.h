#ifndef MPIL_INFO_H
#define MPIL_INFO_H

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Struct for containing number of initialized columns and overall size
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