include_directories(".")

add_subdirectory(utils)
add_subdirectory(locality)
add_subdirectory(persistent)
add_subdirectory(collective)
add_subdirectory(neighborhood)
if(USE_GPU)
    add_subdirectory(heterogeneous)
endif()

add_library(mpi_advance STATIC
    ${utils_SOURCES} ${utils_HEADERS}
    ${locality_SOURCES} ${locality_HEADERS}
    ${collective_SOURCES} ${collective_HEADERS}
    ${persistent_SOURCES} ${persistent_HEADERS}
    ${neighborhood_SOURCES} ${neighborhood_HEADERS})

target_link_libraries(mpi_advance ${MPI_LIBRARIES} stdc++)

install(TARGETS mpi_advance DESTINATION "lib")
install(FILES mpi_advance.h DESTINATION "include/src")
install(FILES ${utils_HEADERS} DESTINATION "include/src/utils")
install(FILES ${locality_HEADERS} DESTINATION "include/src/locality")
install(FILES ${collective_HEADERS} DESTINATION "include/src/collective")
install(FILES ${neighborhood_HEADERS} DESTINATION "include/src/neighborhood")
install(FILES ${persistent_HEADERS} DESTINATION "include/src/persistent")
if (USE_GPU)
    install(FILES ${heterogeneous_HEADERS} DESTINATION "include/src/heterogeneous")
endif()

if (ENABLE_UNIT_TESTS)
    enable_testing()
    add_subdirectory(collective/tests)
    add_subdirectory(neighborhood/tests)
endif()

