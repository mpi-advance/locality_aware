cmake_minimum_required(VERSION 3.15)
project(openshmem_alltoallv C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# finding the OpenSHMEM compiler, wrapper used for this benchmark
find_program(OSHCC oshcc)
if (NOT OSHCC)
  message(FATAL_ERROR "oshcc not found. Load your OpenSHMEM tools:) (e.g., module load openmpi/4.1.2
,module load gcc/11.2.1) and retry.Note: this may work for Dane and im not sure about other systems ")
endif()
message(STATUS "Using OpenSHMEM compiler wrapper: ${OSHCC}")
set(CMAKE_C_COMPILER ${OSHCC})


include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SRC_FILES src/*.c)


add_library(osh_a2av STATIC ${SRC_FILES})
target_include_directories(osh_a2av PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Building the benchmark
add_executable(bench_alltoallv bench/bench_alltoallv.c)
target_link_libraries(bench_alltoallv PRIVATE osh_a2av oshmem)


