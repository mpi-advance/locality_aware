enable_testing()

## Grab all .cpp files in test directory
file(GLOB CPP_SOURCES CONFIGURE_DEPENDS "*.cpp" ".c")

## All should be compiled appropriately (cxx vs nvcc vs hipcc)
#set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE ${locality_aware_LANG})
foreach(src ${CPP_SOURCES})
	make_test(${src} 16)
endforeach()

if(USE_CUDA)
	add_subdirectory(gpu_tests)
endif()
