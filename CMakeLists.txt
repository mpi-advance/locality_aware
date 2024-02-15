cmake_minimum_required(VERSION 3.14 FATAL_ERROR) #3.14 required for new version of googletest
project(mpi_advance VERSION 2.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

## User customizable build options
option(ENABLE_UNIT_TESTS "Enable unit testing" OFF)
set(MPI_ADVANCE_ALLGATHER_MODE  "default" CACHE STRING "The underlying algorithm for MPI Advance MPI_Allgather wrapper")
set(MPI_ADVANCE_XALLGATHER_MODE "default" CACHE STRING "The underlying algorithm for MPIX_Allgather")
set(MPI_ADVANCE_ALLTOALL_MODE   "default" CACHE STRING "The underlying algorithm for MPI Advance MPI_Alltoall wrapper")
set(MPI_ADVANCE_XALLTOALL_MODE  "default" CACHE STRING "The underlying algorithm for MPIX_Alltoall")

set(MPI_ADVANCE_ALLGATHER_MODES "bruck;p2p;ring;")
set_property(CACHE MPI_ADVANCE_ALLGATHER_MODE PROPERTY STRINGS ${MPI_ADVANCE_ALLGATHER_MODES})

set(MPI_ADVANCE_XALLGATHER_MODES "bruck;p2p;ring;heir_bruck;mult_heir_bruck")
set_property(CACHE MPI_ADVANCE_XALLGATHER_MODE PROPERTY STRINGS ${MPI_ADVANCE_XALLGATHER_MODES})

set(MPI_ADVANCE_ALLTOALL_MODES "pairwise;bruck")
set_property(CACHE MPI_ADVANCE_ALLTOALL_MODE PROPERTY STRINGS ${MPI_ADVANCE_ALLTOALL_MODES})

set(MPI_ADVANCE_XALLTOALL_MODES "pairwise")
set_property(CACHE MPI_ADVANCE_XALLTOALL_MODE PROPERTY STRINGS ${MPI_ADVANCE_XALLTOALL_MODES})

## Process user options
# Allgather mode:
string( TOLOWER "${MPI_ADVANCE_ALLGATHER_MODE}" MPI_ADVANCE_ALLGATHER_MODE )
if(MPI_ADVANCE_ALLGATHER_MODE IN_LIST MPI_ADVANCE_ALLGATHER_MODES)
  add_definitions(-DMPI_ADVANCE_ALLGATHER_${MPI_ADVANCE_ALLGATHER_MODE})
  message(STATUS "Using the ${MPI_ADVANCE_ALLGATHER_MODE} MPI Advanced MPI_Allgather mode.")
else()
  message(STATUS "No MPI_ADVANCE_ALLGATHER_MODE provided, defaulting to PMPI_Allgather")
endif()
# XAllgather mode:
string( TOLOWER "${MPI_ADVANCE_XALLGATHER_MODE}" MPI_ADVANCE_XALLGATHER_MODE )
if(MPI_ADVANCE_XALLGATHER_MODE IN_LIST MPI_ADVANCE_XALLGATHER_MODES)
  add_definitions(-DMPI_ADVANCE_ALLGATHER_${MPI_ADVANCE_XALLGATHER_MODE})
  message(STATUS "Using the ${MPI_ADVANCE_XALLGATHER_MODE} MPI Advanced MPIX_Allgather mode.")
else()
  message(STATUS "No MPI_ADVANCE_XALLGATHER_MODE provided, defaulting to locality p2p mode.")
endif()
# Alltoall mode:
string( TOLOWER "${MPI_ADVANCE_ALLTOALL_MODE}" MPI_ADVANCE_ALLTOALL_MODE )
if(MPI_ADVANCE_ALLTOALL_MODE IN_LIST MPI_ADVANCE_ALLTOALL_MODES)
  add_definitions(-DMPI_ADVANCE_ALLTOALL_${MPI_ADVANCE_ALLTOALL_MODE})
  message(STATUS "Using the ${MPI_ADVANCE_ALLTOALL_MODE} MPI Advanced MPI_Alltoall mode.")
else()
  message(STATUS "No MPI_ADVANCE_ALLTOALL_MODE provided, defaulting PMPI_Alltoall.")
endif()
# XAlltoall mode:
string( TOLOWER "${MPI_ADVANCE_XALLTOALL_MODE}" MPI_ADVANCE_XALLTOALL_MODE )
if(MPI_ADVANCE_XALLTOALL_MODE IN_LIST MPI_ADVANCE_XALLTOALL_MODES)
  add_definitions(-DMPI_ADVANCE_XALLTOALL_${MPI_ADVANCE_XALLTOALL_MODE})
  message(STATUS "Using the ${MPI_ADVANCE_XALLTOALL_MODE} MPI Advanced MPIX_Alltoall mode.")
else()
  message(STATUS "No MPI_ADVANCE_XALLTOALL_MODE provided, defaulting to pairwise locality mode.")
endif()
## end processing user options

find_package(MPI REQUIRED)
if (MPI_VERSION EQUAL 3)
    add_definitions(-DUSING_MPI3)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

##################### 
## GOOGLETEST      ##
#####################
if(ENABLE_UNIT_TESTS)

  include(FetchContent)

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    # Specify the commit you depend on and update it regularly.
    GIT_TAG 58d77fa8070e8cec2dc1ed015d66b454c8d78850
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()
#####################

include_directories(src)
add_subdirectory(src)

add_subdirectory(benchmarks)