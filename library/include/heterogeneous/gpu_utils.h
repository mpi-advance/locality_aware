#ifndef MPIL_GPU_UTILS_H
#define MPIL_GPU_UTILS_H

#ifdef HIP
#include "utils_hip.h"
#endif

#ifdef CUDA
#include "utils_cuda.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Prints error message if ierr is not gpuSuccess **/
void gpu_check(int ierr);


#ifdef __cplusplus
}
#endif

#endif
