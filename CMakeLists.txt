cmake_minimum_required(VERSION 3.14) #3.14 required for new version of googletest
project(mpi_advance)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

enable_language(C)
enable_language(CXX)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_FLAGS "-Wall -Wextra -Wpedantic -Wshadow")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wshadow")
#set(CMAKE_CXX_FLAGS "-fsanitize=address")


option(ENABLE_UNIT_TESTS "Enable unit testing" ON)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)

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

include_directories("src")
add_subdirectory("src")

add_subdirectory(benchmarks)
