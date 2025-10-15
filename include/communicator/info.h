#ifndef MPIL_INFO_X_H
#define MPIL_INFO_X_H

#ifdef __cplusplus
extern "C" {
#endif

// typedef struct _MPIL_Info
// {
//     int crs_num_initialized;
//     int crs_size_initialized;
// } MPIL_Info;

int MPIL_Info_init(MPIL_Info** info);

int MPIL_Info_free(MPIL_Info** info);

#ifdef __cplusplus
}
#endif

#endif
