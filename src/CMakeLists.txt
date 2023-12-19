include_directories(".")

add_subdirectory(locality)
add_subdirectory(persistent)
add_subdirectory(collective)
add_subdirectory(neighborhood)

add_library(mpi_advance STATIC
    utils.h utils.cpp
    ${locality_SOURCES} ${locality_HEADERS}
    ${collective_SOURCES} ${collective_HEADERS}
    ${persistent_SOURCES} ${persistent_HEADERS}
    ${neighborhood_SOURCES} ${neighborhood_HEADERS})

target_link_libraries(mpi_advance ${MPI_LIBRARIES} stdc++)

string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
if(build_type STREQUAL "debug")
	# add additional debug flags for DEBUG
	message(STATUS "Adding flags for debug build")
	target_compile_options(mpi_advance PRIVATE -Wall -Wextra )
    target_compile_options(mpi_advance PRIVATE -Wno-unused-parameter)
endif()

install(TARGETS mpi_advance DESTINATION "lib")
install(FILES mpi_advance.h utils.h DESTINATION "include")
install(FILES ${locality_HEADERS} DESTINATION "include/locality")
install(FILES ${collective_HEADERS} DESTINATION "include/collective")
install(FILES ${neighborhood_HEADERS} DESTINATION "include/neighborhood")
install(FILES ${persistent_HEADERS} DESTINATION "include/persistent")

if (ENABLE_UNIT_TESTS)
    enable_testing()
    add_subdirectory(collective/tests)
    add_subdirectory(neighborhood/tests)
endif()

