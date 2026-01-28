## Grab all .cpp files in test directory
file(GLOB CPP_SOURCES CONFIGURE_DEPENDS "*.cpp" ".c")

foreach(src ${CPP_SOURCES})
#	make_test(${src})
endforeach()

target_include_directories(locality_aware PUBLIC 
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tests>)


if(USE_GPU)
	add_subdirectory(gpu_tests)
endif()
