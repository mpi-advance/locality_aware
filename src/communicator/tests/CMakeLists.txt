# Get all .cpp files in the test direcotry
file(GLOB CPP_SOURCES CONFIGURE_DEPENDS "*.cpp")

set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE ${locality_aware_LANG})

foreach(src ${CPP_SOURCES})

    get_filename_component(exec_name ${src} NAME_WE)
    add_executable(${exec_name} ${src})
    target_link_libraries(${exec_name} mpi_advance ${MPI_LIBRARIES})
    add_test(NAME ${exec_name}_Test COMMAND ${MPIRUN} -n 16 ./${exec_name})
endforeach()