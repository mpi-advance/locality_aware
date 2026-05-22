target_include_directories(locality_aware PRIVATE 
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
add_subdirectory(bindings) 
add_subdirectory(source)

if (ENABLE_UNIT_TESTS)
	add_subdirectory(tests)
endif()
