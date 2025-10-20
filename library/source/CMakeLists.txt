add_subdirectory(collective)
add_subdirectory(communicator)
add_subdirectory(neighborhood)
add_subdirectory(persistent)


if(USE_GPU)
	add_subdirectory(heterogeneous)
endif()


