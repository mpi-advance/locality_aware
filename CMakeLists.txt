cmake_minimum_required(VERSION 2.8)
project(mpi_advance)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

enable_language(C)
enable_language(CXX)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

option(ENABLE_UNIT_TESTS "Enable unit testing" ON)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

if (MPI_VERSION EQUAL 3)
    add_definitions(-DUSING_MPI3)
endif()

#/////////////////////////// star information of google test ///////////////////////////////
set(GOOGLETEST_ROOT external/googletest CACHE STRING "Google Test source root")
#MESSAGE( STATUS "GOOGLETEST_ROOT: "    ${GOOGLETEST_ROOT} )

include_directories(
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/include
    )

#MESSAGE( STATUS "PROJECT_SOURCE_DIR: "    ${PROJECT_SOURCE_DIR} )

set(GOOGLETEST_SOURCES
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/src/gtest-all.cc
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/src/gtest_main.cc
    )


foreach(_source ${GOOGLETEST_SOURCES})
    set_source_files_properties(${_source} PROPERTIES GENERATED 1)
    #MESSAGE( STATUS "_source: "                 ${_source} )
endforeach()


add_library(googletest STATIC ${GOOGLETEST_SOURCES})
#///////////////////////////// end information of google test //////////////////////////////

include_directories(".")
add_subdirectory("src")

