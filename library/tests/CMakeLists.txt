## Grab all .cpp files in test directory
file(GLOB CPP_SOURCES CONFIGURE_DEPENDS "*.cpp" ".c")

foreach(src ${CPP_SOURCES})
	make_test(${src} 16)
endforeach()

if(USE_CUDA)
	add_subdirectory(gpu_tests)
endif()
