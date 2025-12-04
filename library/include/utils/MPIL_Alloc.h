#ifndef MPIL_ALLOC_H
#define MPIL_ALLOC_H

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*MPIL_Alloc_ftn)(void**, const int);
typedef int (*MPIL_Free_ftn)(void*);

#ifdef __cplusplus
}
#endif

#endif
