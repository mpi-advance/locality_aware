cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
cmake_policy(VERSION 3.21.3...3.27)

# Create project
project(locality_aware LANGUAGES C CXX CUDA)

# Set C,CXX standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

# Add warning flags to help with debugging
set(CMAKE_C_FLAGS "-Wall -Wextra -Wpedantic -Wshadow")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wshadow")

# Build Options
option(USE_CUDA "Enable NVIDIA CUDA support" OFF)
option(USE_HIP "Enable AMD HIP support" OFF)
option(GPU_AWARE "Use GPU-Aware MPI, if GPU support is enabled" ON)
option(ENABLE_UNIT_TESTS "Enable unit testing" ON)

# Set default language of .cpp files
set(locality_aware_LANG CXX)

include_directories(.)
include_directories(src)
add_subdirectory(src)

if (ENABLE_UNIT_TESTS)
    enable_testing()
endif()

# Find MPI
find_package(MPI REQUIRED)
if (MPI_COMPILER_FLAGS)
  add_compile_options(${MPI_COMPILER_FLAGS})
endif()
set(MPICXX "mpicxx" CACHE STRING "MPICXX command")
set(CUDA_ARCH "sm_70" CACHE STRING "CUDA Architecture")

# Find GPU Compiler
if (USE_CUDA OR USE_HIP)
  if (GPU_AWARE)
      add_definitions(-DGPU_AWARE)
      message(STATUS "Code will be compiled assuming GPU-Aware support.  If your compiler doesn't not support this, set GPU_AWARE to OFF")
  endif()
  set(USE_GPU ON)  
  add_definitions(-DCUDA)
  add_definitions(-DGPU)

  if (USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "-ccbin=${MPICXX} -arch=${CUDA_ARCH}")
    set(locality_aware_LANG CUDA)
    message(STATUS "CUDA support enabled: ${CMAKE_CUDA_FLAGS}")

  elseif (USE_HIP)
    enable_language(HIP)
    set(locality_aware_LANG HIP)
    message(STATUS "HIP support enabled")

  endif()
  
endif()

# Find OpenMP (not required)
find_package(OpenMP)

set(mpi_advance_INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(MPIRUN "mpirun" CACHE STRING "MPIRUN command")

file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS *.cpp)
set_source_files_properties(
    ${CPP_SOURCES} 
    PROPERTIES LANGUAGE ${locality_aware_LANG})




