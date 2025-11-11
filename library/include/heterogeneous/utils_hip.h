#ifndef UTILS_HIP_HPP
#define UTILS_HIP_HPP

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif

#include "hip/hip_runtime.h"

// Devices
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuSetDevice hipSetDevice

// Data allocation
#define gpuMallocHost hipHostMalloc
#define gpuMalloc hipMallocHost     //--deprecated
#define gpuFree hipFree
#define gpuFreeHost hipHostFree

// Error Handling
#define gpuError hipError_t
#define gpuGetLastError hipGetLastError
#define gpuSuccess hipSuccess

// Memcpy
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyKind hipMemcpyKind
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyHostToHost hipMemcpyHostToHost

// Streams
#define gpuStream_t hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy

// Synchronization
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuStreamSynchronize hipStreamSynchronize

#define gpuMemoryTypeHost hipMemoryTypeHost
#define gpuErrorInvalidValue hipErrorInvalidValue
#define gpuPointerGetAttributes hipPointerGetAttributes
#define gpuMemoryType hipMemoryType
#define gpuMemoryTypeHost hipMemoryTypeHost
#define gpuMemoryTypeDevice hipMemoryTypeDevice
#define gpuPointerAttributes hipPointerAttribute_t
#define gpuMemset hipMemset

#endif
