add_subdirectory(collective)
add_subdirectory(communicator)
add_subdirectory(neighborhood)
add_subdirectory(persistent)
add_subdirectory(utils)



if(USE_CUDA OR USE_HIP)
	add_subdirectory(heterogeneous)
	set_target_properties(locality_aware PROPERTIES FORCE_LANGUAGE CUDA)
endif()


