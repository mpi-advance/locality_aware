## Grab all .cpp files in test directory
file(GLOB CPP_SOURCES CONFIGURE_DEPENDS "*.cpp" "*.c")

foreach(src ${CPP_SOURCES})
	make_executable(${src} TRUE)
endforeach()

if(USE_GPU)
	add_subdirectory(gpu_tests)
endif()
