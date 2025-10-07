add_subdirectory(bindings)

add_subdirectory(collective)
add_subdirectory(communicator)


add_subdirectory(neighborhood)
add_subdirectory(persistent)
add_subdirectory(utils)



if(USE_CUDA OR USE_HIP)
	add_subdirectory(heterogeneous)
	set_target_properties(locality_aware PROPERTIES FORCE_LANGUAGE CUDA)
endif()


set_source_files_properties(
    ${utils_SOURCES}
    ${communicator_SOURCES}
    ${collective_SOURCES}
    ${persistent_SOURCES}
    ${neighborhood_SOURCES}
    ${heterogeneous_SOURCES}
    PROPERTIES LANGUAGE ${locality_aware_LANG})
	
