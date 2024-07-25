include_directories(.)

if (USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

add_subdirectory(utils)
add_subdirectory(locality)
add_subdirectory(persistent)
add_subdirectory(collective)
add_subdirectory(neighborhood)
if(USE_GPU)
    add_subdirectory(heterogeneous)
endif()

if(USE_CUDA)
    set_source_files_properties(
        ${utils_SOURCES} 
        ${locality_SOURCES} 
        ${collective_SOURCES}
        ${persistent_SOURCES} 
        ${neighborhood_SOURCES} 
        ${heterogeneous_SOURCES}
        PROPERTIES LANGUAGE CUDA)
endif()

add_library(mpi_advance
    ${utils_SOURCES} ${utils_HEADERS}
    ${locality_SOURCES} ${locality_HEADERS}
    ${collective_SOURCES} ${collective_HEADERS}
    ${persistent_SOURCES} ${persistent_HEADERS}
    ${neighborhood_SOURCES} ${neighborhood_HEADERS}
    ${heterogeneous_SOURCES} ${heterogeneous_HEADERS}
)

target_link_libraries(mpi_advance ${MPI_LIBRARIES} ${EXTERNAL_LIBS})

if (USE_CUDA)
    target_link_libraries(mpi_advance CUDA::cudart)
endif()

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
if (USE_GPU)
    add_subdirectory(heterogeneous/tests)
endif(USE_GPU)
	#    enable_testing()
	#    add_subdirectory(collective/tests)
	#add_subdirectory(neighborhood/tests)
endif()

