target_sources(locality_aware PRIVATE global_defaults.c)

target_sources(locality_aware PRIVATE Collective/MPIL_Alltoall.c)
target_sources(locality_aware PRIVATE Collective/MPIL_Alltoallv.c)

target_sources(locality_aware PRIVATE CRS/MPIL_Alltoall_crs.c)
target_sources(locality_aware PRIVATE CRS/MPIL_Alltoallv_crs.c)

target_sources(locality_aware PRIVATE MPIL_Alloc/MPIL_Alloc.cpp)
target_sources(locality_aware PRIVATE MPIL_Alloc/MPIL_Free.cpp)

target_sources(locality_aware PRIVATE MPIL_Info/MPIL_Info_init.cpp)
target_sources(locality_aware PRIVATE MPIL_Info/MPIL_Info_free.cpp)

target_sources(locality_aware PRIVATE MPIL_Request/MPIL_Start.c)
target_sources(locality_aware PRIVATE MPIL_Request/MPIL_Wait.c)
target_sources(locality_aware PRIVATE MPIL_Request/MPIL_Request_free.c)

#for directories with several files 
add_subdirectory(Neighborhood)
add_subdirectory(MPIL_Comm)

#add any sources that utilize GPU specfic commands to global list for later conversion. 
if(USE_CUDA)
	get_property(CUDA_SOURCES GLOBAL PROPERTY CUDA_SOURCES_GLOBAL)
	set_property(GLOBAL APPEND PROPERTY CUDA_SOURCES_GLOBAL  "${CMAKE_CURRENT_SOURCE_DIR}/Collective/MPIL_Alltoall.c")
	set_property(GLOBAL APPEND PROPERTY CUDA_SOURCES_GLOBAL  "${CMAKE_CURRENT_SOURCE_DIR}/Collective/MPIL_Alltoallv.c")
	set_property(GLOBAL APPEND PROPERTY CUDA_SOURCES_GLOBAL  "${CMAKE_CURRENT_SOURCE_DIR}/MPIL_Comm/MPIL_Comm_device_init.c")
	set_property(GLOBAL APPEND PROPERTY CUDA_SOURCES_GLOBAL  "${CMAKE_CURRENT_SOURCE_DIR}/MPIL_Comm/MPIL_Comm_device_free.c")
	set_property(GLOBAL APPEND PROPERTY CUDA_SOURCES_GLOBAL  "${CMAKE_CURRENT_SOURCE_DIR}/MPIL_Request/MPIL_Request_free.c")
endif()

