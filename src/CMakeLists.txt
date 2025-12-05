include_directories(".")

add_subdirectory(utils)
add_subdirectory(communicator)
add_subdirectory(persistent)
add_subdirectory(collective)
add_subdirectory(neighborhood)
if(USE_CUDA OR USE_HIP)
    add_subdirectory(heterogeneous)
endif()

set_source_files_properties(
    ${utils_SOURCES}
    ${communicator_SOURCES}
    ${collective_SOURCES}
    ${persistent_SOURCES}
    ${neighborhood_SOURCES}
    ${heterogeneous_SOURCES}
    PROPERTIES LANGUAGE ${locality_aware_LANG})

add_library(mpi_advance
    ${utils_SOURCES} ${utils_HEADERS}
    ${communicator_SOURCES} ${communicator_HEADERS}
    ${collective_SOURCES} ${collective_HEADERS}
    ${persistent_SOURCES} ${persistent_HEADERS}
    ${neighborhood_SOURCES} ${neighborhood_HEADERS}
    ${heterogeneous_SOURCES} ${heterogeneous_HEADERS}
)

target_link_libraries(mpi_advance ${MPI_LIBRARIES})

message(STATUS "Unit tests enabled: ${ENABLE_UNIT_TESTS}")
if (ENABLE_UNIT_TESTS)
    add_subdirectory(collective/tests)
    add_subdirectory(communicator/tests)
    add_subdirectory(neighborhood/tests)
    if (USE_GPU)
        add_subdirectory(heterogeneous/tests)
    endif(USE_GPU)
endif()

