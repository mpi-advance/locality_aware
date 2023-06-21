cmake_minimum_required(VERSION 3.17 FATAL_ERROR) 

include(FeatureSummary)

enable_language(C)
enable_language(CXX)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Options for Heterogeneous Architectures
option(GPU_AWARE "Use GPU-Aware MPI" ON)
option(USE_GPU "Compile heterogeneous folder" OFF)
option(USE_HIP "Compile with HIP Support" OFF)
option(USE_CUDA "Compile with CUDA Support" OFF)
option(ENABLE_UNIT_TESTS "Enable unit testing" ON)

add_feature_info(gpu_aware GPU_AWARE "Use GPU-Aware MPI")
add_feature_info(use_hip USE_HIP "Compile with HIP Support")
add_feature_info(use_cuda USE_CUDA "Compile with CUDA Support")


set(MPIRUN "mpirun" CACHE STRING "MPIRUN command")
set(MPICXX "mpicxx" CACHE STRING "MPICXX command")

if (USE_CUDA)
    project(mpi_advance LANGUAGES CUDA CXX VERSION 0.1.0)
elseif (USE_HIP)
    project(mpi_advance LANGUAGES HIP CXX VERSION 0.1.0)
else()
    project(mpi_advance)
endif()
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)


find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(mpi_advance_INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

cmake_policy(SET CMP0104 OLD)

#execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
#                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#                        RESULT_VARIABLE GIT_SUBMOD_RESULT)

if (USING_AMD)
    set(USE_HIP ON)
    set(USE_CUDA OFF)
    set(ATDM_MPI_LIB "mpi_amd")
    set(MPICH_GPU_SUPPORT_ENABLED 1)
    include_directories($ENV{ROCM_PATH}/include)
    set(CMAKE_EXE_LINKER_FLAGS "-L$ENV{ROCM_PATH}/lib -Wl,-rpath,$ENV{ROCM_PATH}/lib -lamdhip64 -L$ENV{CRAY_MPICH_DIR}/lib -Wl,-rpath,$ENV{CRAY_MPICH_DIR}/lib -l$ENV{ATDM_MPI_LIB} -L$ENV{CRAY_MPICH_ROOTDIR}/gtl/lib -Wl,-rpath,$ENV{CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa -L$ENV{ROCM_PATH}/llvm/lib/clang/15.0.0/lib/linux/ -lclang_rt.builtins-x86_64 -x none")
    set(CMAKE_SHARED_LINKER_FLAGS "-x none")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --rocm-path=$ROCM_PATH -D__HIP_PLATFORM_AMD__")
endif(USING_AMD)

if (USE_CUDA)
    if (GPU_AWARE)
        add_definitions(-DGPU_AWARE)
    endif()
    set(USE_GPU ON)
    set(CMAKE_CUDA_FLAGS "-ccbin=${MPICXX} -arch=sm_70")
    add_definitions(-DCUDA)
    add_definitions(-DGPU)
endif(USE_CUDA)

if (USE_HIP)
    if (GPU_AWARE)
        add_definitions(-DGPU_AWARE)
    endif()
    set(USE_GPU ON)
    add_definitions(-DHIP)
    add_definitions(-DGPU)
endif(USE_HIP)


##################### 
## GOOGLETEST      ##
#####################
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
#####################

include_directories(.)
include_directories(src)
add_subdirectory(src)
if (USE_CUDA)
    target_compile_options(mpi_advance PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: # Flags for CUDA Objects
    -lineinfo
    -Xcompiler -fopenmp
    >
    $<$<COMPILE_LANGUAGE:CXX>: # Flags for C++ Objects
    -fno-omit-frame-pointer
    -fopenmp
    >
    )
    set_property(TARGET mpi_advance PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif(USE_CUDA)

add_subdirectory(benchmarks)

