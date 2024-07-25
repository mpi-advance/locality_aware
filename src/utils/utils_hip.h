#ifndef UTILS_HIP_HPP
#define UTILS_HIP_HPP

#include "hip/hip_runtime_api.h"

// Devices
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuSetDevice hipSetDevice

// Data allocation
#define gpuMallocHost hipHostMalloc
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuFreeHost hipHostFree

// Error Handling
#define gpuError hipError_t
#define gpuGetLastError hipGetLastError
#define gpuSuccess hipSuccess

// Memcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyKind hipMemcpyKind
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice

// Streams
#define gpuStream_t hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy 

// Synchronization
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuStreamSynchronize hipStreamSynchronize

#endif
