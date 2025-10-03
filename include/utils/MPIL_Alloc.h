#ifndef MPIL_ALLOC_H
#define MPIL_ALLOC_H

#ifdef __cplusplus
extern "C" {
#endif

// Allocate Vector in MPI
int MPIL_Alloc(void** pointer, const int bytes);
int MPIL_Free(void* pointer);

#ifdef __cplusplus
}
#endif

#endif