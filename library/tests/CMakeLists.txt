enable_testing()

## Grab all .cpp files in test directory
file(GLOB CPP_SOURCES CONFIGURE_DEPENDS "*.cpp")

## All should be compiled appropriately (cxx vs nvcc vs hipcc)
set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE ${locality_aware_LANG})

## Go through each CPP file
foreach(src ${CPP_SOURCES})
    
    ## Grab the name without full path or extension
    get_filename_component(exec_name ${src} NAME_WE)

    ## Create executable
    add_executable(${exec_name} ${src})

    # Link with MPI 
    target_link_libraries(${exec_name} locality_aware ${MPI_LIBRARIES})

    # Add to CTEST
    add_test(NAME ${exec_name}_Test COMMAND ${MPIRUN} -n 4 ./${exec_name})
endforeach()


