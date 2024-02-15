cmake_minimum_required(VERSION 3.14 FATAL_ERROR) #3.14 required for new version of googletest
project(mpi_advance VERSION 2.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_UNIT_TESTS "Enable unit testing" OFF)

# A helper function to handle processing of the various collective modes in MPI Advance
# collective_name = the name of the collective, case-insensitive (e.g Alltoall)
# var_description = the description to be displayed with the associated cmake option
# var_options     = A cmake list containing valid options for choices for the collective algorithms
# default_mode    = Only used for print outs, to say what the default algorithm is if not specified.
# 
# Creates the following variable that users can specify from command line:
# -> "MPI_ADVANCE_${uc_name}_MODE" where:
#     ${uc_name} is the capitalized version of ${collective_name}
# For example, if "add_mpi_advance_option(BCast ... ... ...)" is called in the cmake code,
# the user can do "-DMPI_ADVANCE_BCAST_MODE=YYY" to select the YYY algorithm for "BCast"
#
# If the created cmake variable is set, the the following defines is added to the compile:
# -> "-DMPI_ADVANCE_${uc_name}_{lower_name}" where:
#    ${uc_name} is the capitalized version of ${collective_name}
#    ${collective_name} is the lower case version of the variable selected by the user
#
function(add_mpi_advance_option collective_name var_description var_options default_mode)
  string(TOUPPER ${collective_name} uc_name)
  set(var_name "MPI_ADVANCE_${uc_name}_MODE")
  set(${var_name} "default" CACHE STRING ${var_description})

  set(MA_COLLECTIVE_OPTIONS ${var_options})
  set_property(CACHE ${var_name} PROPERTY STRINGS MA_COLLECTIVE_OPTIONS)

  string(TOLOWER ${${var_name}} lower_name )

  if(lower_name IN_LIST MA_COLLECTIVE_OPTIONS)
    add_definitions(-DMPI_ADVANCE_${uc_name}_${lower_name})
    message(STATUS "Using the ${lower_name} for ${var_name}.")
  else()
    message(STATUS "No ${var_name} provided, defaulting to ${default_mode}.")
  endif()
endfunction()


# Add all the collectives that have options here
add_mpi_advance_option(ALLGATHER
                       "The underlying algorithm for MPI Advance MPI_Allgather wrapper"
                       "bruck;p2p;ring;"
                       "PMPI_Allgather")

add_mpi_advance_option(XALLGATHER
                       "The underlying algorithm for MPIX_Allgather"
                       "bruck_locality;p2p_locality;ring_locality;heir_bruck;mult_heir_bruck"
                       "p2p")

add_mpi_advance_option(ALLTOALL
                       "The underlying algorithm for MPI Advance MPI_Alltoall wrapper"
                       "pairwise;bruck"
                       "PMPI_Alltoall")

add_mpi_advance_option(XALLTOALL
                       "The underlying algorithm for MPIX_Alltoall"
                       "pairwise_locality"
                       "pairwise_locality")

add_mpi_advance_option(ALLTOALLV
                       "The underlying algorithm for MPI Advance MPI_Alltoallv wrapper"
                       "pairwise;nonblocking;pairwise_nonblocking;waitany;pairwise_nonblocking_log2"
                       "PMPI_Alltoallv")

add_mpi_advance_option(XALLTOALLV
                       "The underlying algorithm for MPIX_Alltoallv"
                       "pairwise_locality"
                       "pairwise_locality")

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