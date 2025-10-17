target_sources(locality_aware PRIVATE 
	global_defaults.c
	Collective/MPIL_Alltoall.c
	Collective/MPIL_Alltoallv.c
	Collective/MPIL_Alltoall_crs.c
	Collective/MPIL_Alltoallv_crs.c
	MPIL_Alloc/MPIL_Alloc.cpp
	MPIL_Alloc/MPIL_Free.cpp
	MPIL_Info/MPIL_Info_init.cpp
	MPIL_Info/MPIL_Info_free.cpp
	MPIL_Request/MPIL_Start.c
	MPIL_Request/MPIL_Wait.c
	MPIL_Request/MPIL_Request_free.c
)

#for directories with several files 
add_subdirectory(Neighborhood)
add_subdirectory(MPIL_Comm)

#add any sources that utilize GPU specific commands to global list for later conversion. 
if(USE_GPU)
	get_property(GPU_SOURCES GLOBAL PROPERTY GPU_SOURCES_GLOBAL)
	set_property(GLOBAL APPEND PROPERTY GPU_SOURCES_GLOBAL  
		"${CMAKE_CURRENT_SOURCE_DIR}/Collective/MPIL_Alltoall.c"
		"${CMAKE_CURRENT_SOURCE_DIR}/Collective/MPIL_Alltoallv.c"
		"${CMAKE_CURRENT_SOURCE_DIR}/Collective/MPIL_Alltoall_crs.c"
		"${CMAKE_CURRENT_SOURCE_DIR}/Collective/MPIL_Alltoallv_crs.c"
		"${CMAKE_CURRENT_SOURCE_DIR}/MPIL_Comm/MPIL_Comm_device_init.c"
		"${CMAKE_CURRENT_SOURCE_DIR}/MPIL_Comm/MPIL_Comm_device_free.c"
		"${CMAKE_CURRENT_SOURCE_DIR}/MPIL_Request/MPIL_Request_free.c"
	)
endif()

