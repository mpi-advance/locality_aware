include_directories(".")

add_subdirectory(utils)
add_subdirectory(locality)
add_subdirectory(persistent)
add_subdirectory(collective)
add_subdirectory(neighborhood)
if(USE_GPU)
    add_subdirectory(heterogeneous)
endif()

message(STATUS ${CMAKE_C_FLAGS})
add_library(mpi_advance
    ${utils_SOURCES} ${utils_HEADERS}
    ${locality_SOURCES} ${locality_HEADERS}
    ${collective_SOURCES} ${collective_HEADERS}
    ${persistent_SOURCES} ${persistent_HEADERS}
    ${neighborhood_SOURCES} ${neighborhood_HEADERS}
    ${heterogeneous_SOURCES} ${heterogeneous_HEADERS}
)

#target_link_libraries(mpi_advance ${MPI_LIBRARIES} ${EXTERNAL_LIBS} OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
target_link_libraries(mpi_advance ${MPI_LIBRARIES})

if (ENABLE_UNIT_TESTS)
    add_subdirectory(collective/tests)
    add_subdirectory(neighborhood/tests)
    if (USE_GPU)
        add_subdirectory(heterogeneous/tests)
    endif(USE_GPU)
endif()

