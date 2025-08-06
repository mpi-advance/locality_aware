cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
cmake_policy(VERSION 3.21.1...3.27)

# Create project
project(locality_aware LANGUAGES C CXX)

# Set C,CXX standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

# Add warning flags to help with debugging
string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE} -> ${build_type}")
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

add_subdirectory(src)

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

set_target_properties(mpi_advance PROPERTIES PUBLIC_HEADER src/mpi_advance.h)

target_include_directories(mpi_advance PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
    )

target_link_libraries(mpi_advance PUBLIC ${EXTERNAL_LIBS})

install(TARGETS mpi_advance)

### Install all headers that locality_aware exports
install(FILES ${utils_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/utils)
install(FILES ${communicator_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/communicator)
install(FILES ${collective_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/collective)
install(FILES ${persistent_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/persistent)
install(FILES ${neighborhood_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/neighborhood)
install(FILES ${heterogeneous_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/heterogeneous)

add_subdirectory(benchmarks)

### Installation Requirements for Spack Package ###
include(CMakePackageConfigHelpers)

# Install the actual target(s) and register them for export
install(TARGETS mpi_advance
        EXPORT mpiadvanceTargets
        DESTINATION lib)

# Export the targets into a CMake package
install(EXPORT mpiadvanceTargets
        NAMESPACE MPIAdvance::
        DESTINATION share/mpiadvance)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mpiadvanceConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/mpiadvance-config.cmake"
  INSTALL_DESTINATION share/mpiadvance)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/mpiadvance-config.cmake"
  DESTINATION share/mpiadvance)
############## End of Spack Section ################

