#always add to gpu language conversion list. 
get_property(GPU_SOURCES GLOBAL PROPERTY GPU_SOURCES_GLOBAL)
set_property(GLOBAL APPEND PROPERTY GPU_SOURCES_GLOBAL  
	${CMAKE_CURRENT_SOURCE_DIR}/get_mem_types.c
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_alltoall.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_alltoallv.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_allgather.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_allreduce.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_check.c
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_repack.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/repack.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/device_repack.cpp 
	${CMAKE_CURRENT_SOURCE_DIR}/get_memcpy_kind.c
)

target_sources(locality_aware PRIVATE 
	get_mem_types.c
    gpu_allgather.cpp
    gpu_allreduce.cpp
    gpu_alltoall.cpp
	gpu_alltoallv.cpp
	gpu_check.c
	gpu_repack.cpp
	repack.cpp
	device_repack.cpp 
	get_memcpy_kind.c
)
