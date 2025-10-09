cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
cmake_policy(VERSION 3.21.1...3.27)

# Create project
project(locality_aware VERSION 2.0.0 LANGUAGES C CXX)

# Set C,CXX standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
	message(STATUS "No CMAKE_BUILD_TYPE specified, setting build type to RELEASE.")
	set(CMAKE_BUILD_TYPE "RELEASE")
endif()

# Convert to lower case string to ignore case
string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE} -> ${build_type}")

# Add warning flags to help with debugging
if(build_type STREQUAL "debug")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -Wshadow")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wshadow")
endif()

# Build Options
option(USE_CUDA "Enable NVIDIA CUDA support" OFF)
option(USE_HIP "Enable AMD HIP support" OFF)
option(GPU_AWARE "Use GPU-Aware MPI, if GPU support is enabled" ON)
option(ENABLE_UNIT_TESTS "Enable unit testing" ON)

set(MPICXX "mpicxx" CACHE STRING "MPICXX command")
set(MPIRUN "mpirun" CACHE STRING "MPIRUN command")
set(CUDA_ARCH "75" CACHE STRING "CUDA Architecture")

# Find MPI
find_package(MPI REQUIRED)
set(EXTERNAL_LIBS "MPI::MPI_CXX")

# Find OpenMP (not required)
find_package(OpenMP)
if (${OpenMP_CXX_FOUND})
  list(APPEND EXTERNAL_LIBS "OpenMP::OpenMP_CXX")
  add_definitions(-DOPENMP)
endif()

# Set default language of .cpp files
set(locality_aware_LANG CXX)

if (USE_CUDA OR USE_HIP)
  if (GPU_AWARE)
      add_definitions(-DGPU_AWARE)
      message(STATUS "Code will be compiled assuming GPU-Aware support.  If your compiler doesn't not support this, set GPU_AWARE to OFF")
  endif()
  set(USE_GPU ON)
  add_definitions(-DGPU)
if (USE_CUDA)
  set(CMAKE_CUDA_HOST_COMPILER "${MPICXX}")
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
  enable_language(CUDA)
  set(locality_aware_LANG CUDA)
  message(STATUS "CUDA support enabled: ${CMAKE_CUDA_FLAGS}")
  add_definitions(-DCUDA)

elseif (USE_HIP)
  set(CMAKE_HIP_COMPILER "${MPICXX}")
  enable_language(HIP)
  set(locality_aware_LANG HIP)
  message(STATUS "HIP support enabled")
  add_definitions(-DHIP)
endif()
endif()

message(STATUS "Compiling all files as: ${locality_aware_LANG}")

if (ENABLE_UNIT_TESTS)
    enable_testing()
endif()

add_library(locality_aware "")
set_target_properties(locality_aware PROPERTIES PUBLIC_HEADER include/locality_aware.h)
add_subdirectory(library)

if (USE_CUDA)
	get_target_property(CUR_SOURCES locality_aware SOURCES)
	foreach(file ${CUR_SOURCES})
		set_source_files_properties(${file} PROPERTIES LANGUAGE CUDA)
	endforeach()
endif()

target_include_directories(locality_aware PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/library>
    $<INSTALL_INTERFACE:include>
    )




target_link_libraries(locality_aware PUBLIC ${EXTERNAL_LIBS})



install(TARGETS locality_aware)


set_source_files_properties(
    ${utils_SOURCES}
    ${communicator_SOURCES}
    ${collective_SOURCES}
    ${persistent_SOURCES}
    ${neighborhood_SOURCES}
    ${heterogeneous_SOURCES}
    PROPERTIES LANGUAGE ${locality_aware_LANG})

# message (STATUS "CHECK")
# foreach(file ${collective_HEADERS})
	# message (STATUS "copied ${file}" )
# endforeach()

install(DIRECTORY include/communicator/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/communicator)
install(DIRECTORY include/collective/   DESTINATION ${CMAKE_INSTALL_PREFIX}/include/collective)
install(DIRECTORY include/persistent/   DESTINATION ${CMAKE_INSTALL_PREFIX}/include/persistent)
install(DIRECTORY include/utils/   DESTINATION ${CMAKE_INSTALL_PREFIX}/include/utils)
install(DIRECTORY include/neighborhood/   DESTINATION ${CMAKE_INSTALL_PREFIX}/include/neighborhood)
#install(DIRECTORY include/heterogeneous/   DESTINATION ${CMAKE_INSTALL_PREFIX}/include/heterogeneous)

# Install all headers that locality_aware exports
#install(FILES ${utils_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/utils)
# install(FILES ${communicator_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/communicator)
# install(FILES ${collective_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/collective)
# install(FILES ${persistent_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/persistent)
# install(FILES ${neighborhood_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/neighborhood)
# install(FILES ${heterogeneous_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/heterogeneous)



### Installation Requirements for Spack Package ###
include(CMakePackageConfigHelpers)

# Install the actual target(s) and register them for export
install(TARGETS locality_aware
        EXPORT locality_awareTargets
        DESTINATION lib)

# Create the configuration file
configure_package_config_file(
	${CMAKE_CURRENT_SOURCE_DIR}/cmake/locality_aware-config.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
	INSTALL_DESTINATION share/${PROJECT_NAME}
)

## Version file
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config-version.cmake
	COMPATIBILITY SameMinorVersion
)

install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config-version.cmake
	${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
	DESTINATION share/${PROJECT_NAME}
)

# Export the targets into a CMake package
install(EXPORT locality_awareTargets
        NAMESPACE MPIAdvance::
        DESTINATION share/${PROJECT_NAME})

############## End of Spack Section ################

