## Make a list for all the benchmarks to compile
set(CPP_SOURCES "")

## Most benchmarks will go in just fine
list(APPEND CPP_SOURCES "alltoall_crs.cpp")
list(APPEND CPP_SOURCES "alltoallv_crs.cpp")
#list(APPEND CPP_SOURCES "microbenchmarks.cpp")
list(APPEND CPP_SOURCES "p2p_alltoall.cpp")
list(APPEND CPP_SOURCES "p2p_alltoallv.cpp")
list(APPEND CPP_SOURCES "allreduce.cpp")

## But some need special requirements -- removed until necessary parameters obtained. 
# if(USE_GPU)
    # list(APPEND CPP_SOURCES "gpu_alltoall.cpp")
# endif()

## All should be compiled appropriately (cxx vs nvcc vs hipcc)
set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE ${locality_aware_LANG})

## Assign a matrix file for testing. 
set(TEST_MATRIX_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../test_data/bcsstk01.pm")

## Go through each CPP file
foreach(src ${CPP_SOURCES})
    make_executable(${src} FALSE)
endforeach()
