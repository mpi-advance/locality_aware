#always add to gpu language conversion list. 
get_property(GPU_SOURCES GLOBAL PROPERTY GPU_SOURCES_GLOBAL)
set_property(GLOBAL APPEND PROPERTY GPU_SOURCES_GLOBAL  
	${CMAKE_CURRENT_SOURCE_DIR}/get_mem_types.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_alltoall.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_allreduce.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_allreduce_init.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_check.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/repack.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/device_repack.cpp 
	${CMAKE_CURRENT_SOURCE_DIR}/get_memcpy_kind.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_alltoallv.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_repack.cpp
)

target_sources(locality_aware PRIVATE 
	get_mem_types.cpp
	gpu_alltoall.cpp
    gpu_allreduce.cpp
    gpu_allgather.cpp
    gpu_allreduce_init.cpp
	gpu_check.cpp
	repack.cpp
	device_repack.cpp 
	get_memcpy_kind.cpp
	gpu_alltoallv.cpp
	gpu_repack.cpp
)
