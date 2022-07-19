include_directories(".")

add_subdirectory(locality)
add_subdirectory(persistent)
add_subdirectory(collective)
add_subdirectory(neighborhood)

add_library(mpi_advance STATIC
    ${locality_SOURCES} ${locality_HEADERS}
    ${collective_SOURCES} ${collective_HEADERS}
    ${persistent_SOURCES} ${persistent_HEADERS}
    ${neighborhood_SOURCES} ${neighborhood_HEADERS})

target_link_libraries(mpi_advance ${MPI_LIBRARIES} stdc++)

install(TARGETS mpi_advance DESTINATION "lib")
install(FILES mpi_advance.h DESTINATION "include/src")
install(FILES ${locality_HEADERS} DESTINATION "include/src/locality")
install(FILES ${collective_HEADERS} DESTINATION "include/src/collective")
install(FILES ${neighborhood_HEADERS} DESTINATION "include/src/neighborhood")

if (ENABLE_UNIT_TESTS)
    enable_testing()
    add_subdirectory(collective/tests)
    add_subdirectory(neighborhood/tests)
endif()

