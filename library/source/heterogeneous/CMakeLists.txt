#always add to gpu language conversion list. 
get_property(GPU_SOURCES GLOBAL PROPERTY GPU_SOURCES_GLOBAL)
set_property(GLOBAL APPEND PROPERTY GPU_SOURCES_GLOBAL  
	${CMAKE_CURRENT_SOURCE_DIR}/get_mem_types.c
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_alltoall.c
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_allreduce.c
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_allreduce_init.c
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_check.c
	${CMAKE_CURRENT_SOURCE_DIR}/repack.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/device_repack.cpp 
	${CMAKE_CURRENT_SOURCE_DIR}/get_memcpy_kind.c
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_alltoallv.c
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_repack.cpp
)

target_sources(locality_aware PRIVATE 
	get_mem_types.c
	gpu_alltoall.c
    gpu_allreduce.c
	gpu_check.c
	repack.cpp
	device_repack.cpp 
	get_memcpy_kind.c
	gpu_alltoallv.c
	gpu_repack.cpp
)
