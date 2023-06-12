#ifndef UTILS_CUDA_HPP
#define UTILS_CUDA_HPP

// Devices
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuSetDevice cudaSetDevice

// Data allocation
#define gpuMallocHost cudaMallocHost
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuFreeHost cudaFreeHost

// Error Handling
#define gpuError cudaError
#define gpuSuccess cudaSuccess
#define gpuGetLastError cudaGetLastError

// Memcpy
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

// Streams
#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy 

// Synchronization
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuStreamSynchronize cudaStreamSynchronize


#endif
