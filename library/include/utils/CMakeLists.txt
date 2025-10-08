
if (USE_CUDA)
	message (STATUS "INCLUDING CUDA HEADER")
	target_sources(locality_aware PRIVATE utils_cuda.h)
elseif (USE_HIP)
	message (STATUS "INCLUDING HIP HEADER")
	target_sources(locality_aware PRIVATE utils_hip.h)
endif()

