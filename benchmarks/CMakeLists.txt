## Make a list for all the benchmarks to compile
set(CPP_SOURCES "")

## Most benchmarks will go in just fine
list(APPEND CPP_SOURCES "alltoall_crs.cpp")
list(APPEND CPP_SOURCES "alltoallv_crs.cpp")
list(APPEND CPP_SOURCES "microbenchmarks.cpp")
list(APPEND CPP_SOURCES "p2p_alltoall.cpp")
list(APPEND CPP_SOURCES "p2p_alltoallv.cpp")
list(APPEND CPP_SOURCES "allreduce_topology.cpp")
list(APPEND CPP_SOURCES "allgather_topology.cpp")
list(APPEND CPP_SOURCES "network_discovery.cpp")

## But some need special requirements
if(USE_GPU)
    list(APPEND CPP_SOURCES "gpu_alltoall.cpp")
endif()

## All should be compiled appropriately (cxx vs nvcc vs hipcc)
set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE ${locality_aware_LANG})

## Go through each CPP file
foreach(src ${CPP_SOURCES})

    ## Grab the name without full path or extension
    get_filename_component(exec_name ${src} NAME_WE)

    ## Create executable
    add_executable(${exec_name} ${src})

    # Link with MPI 
    target_link_libraries(${exec_name} mpi_advance ${MPI_LIBRARIES})
endforeach()
