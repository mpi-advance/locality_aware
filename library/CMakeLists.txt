add_subdirectory(collective)
add_subdirectory(communicator)

if(USE_CUDA OR USE_HIP)
	add_subdirectory(heterogeneous)
endif()
add_subdirectory(neighborhood)
add_subdirectory(persistent)
add_subdirectory(utils)

