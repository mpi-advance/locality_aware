cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
cmake_policy(VERSION 3.21.1...3.27)

# Create project
project(locality_aware VERSION 2.0.0 LANGUAGES C CXX)

# Set C,CXX standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)

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
option(USE_APU "System has APUs" OFF)
option(ENABLE_UNIT_TESTS "Enable unit testing" ON)
option(BENCHMARKS "Enable benchmarks" ON)

set(MPIRUN "mpirun" CACHE STRING "MPIRUN command")

# Find MPI
find_package(MPI REQUIRED)
set(EXTERNAL_LIBS "MPI::MPI_CXX")

# Set default language of .cpp files
set(locality_aware_LANG CXX)
add_library(locality_aware "")

if (USE_CUDA OR USE_HIP)
	if (GPU_AWARE)
	  add_definitions(-DGPU_AWARE)
	  message(STATUS "Code will be compiled assuming GPU-Aware support.  If your compiler doesn't not support this, set GPU_AWARE to OFF")
	endif()
	set(USE_GPU ON)
	target_compile_definitions(locality_aware PUBLIC GPU)
	define_property(GLOBAL PROPERTY CUDA_SOURCES_GLOBAL)
	set_property(GLOBAL PROPERTY GPU_SOURCES_GLOBAL "")
  
	if (USE_CUDA)
	  enable_language(CUDA)
	  set(locality_aware_LANG CUDA)
	  message(STATUS "CUDA support enabled: ${CMAKE_CUDA_FLAGS}")
	  target_compile_definitions(locality_aware PUBLIC CUDA)
	 
	elseif (USE_HIP)
	  find_package(hip REQUIRED)
	  target_link_libraries(locality_aware PUBLIC hip::host)
      target_compile_options(locality_aware PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-x hip>)
	  message(STATUS "HIP support enabled")
	  target_compile_definitions(locality_aware PUBLIC HIP)
	endif()

    if (USE_APU)
        add_definitions(-DAPU)
    endif()
endif()

if (ENABLE_UNIT_TESTS)
    enable_testing()
	set(TEST_PROCS "16" CACHE STRING "Number of processes to use when making ctests")
	function(make_test file)
		#get file name for unique handle
		get_filename_component(exec_name ${file} NAME_WE)
		add_executable(${exec_name} ${file})
		target_link_libraries(${exec_name} locality_aware ${EXTERNAL_LIBS})
		add_test(NAME ${exec_name}_Test COMMAND ${MPIRUN} -n ${TEST_PROCS} ./${exec_name})
	endfunction()
endif()

# Flag == TRUE means TEST
# Flag == FALSE means BENCHMARK
function(make_executable filename flag)
	get_filename_component(exec_name ${filename} NAME_WE)
	add_executable(${exec_name} ${filename})
	target_link_libraries(${exec_name} locality_aware ${EXTERNAL_LIBS})
	
	if(${flag})
		set(locality_test_name ${exec_name}_Test)
	else()
		set(locality_test_name ${exec_name}_Benchmark)
		install(TARGETS ${exec_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
	endif()

	if(ENABLE_UNIT_TESTS)
		add_test(NAME ${locality_test_name} COMMAND ${MPIRUN} -n ${TEST_PROCS} ./${exec_name} ${TEST_MATRIX_FILE})
	endif()

endfunction()

if (BENCHMARKS)
	add_subdirectory(benchmarks)
endif()

set_target_properties(locality_aware PROPERTIES PUBLIC_HEADER include/locality_aware.h)
add_subdirectory(include)
add_subdirectory(library)
target_link_libraries(locality_aware PUBLIC ${EXTERNAL_LIBS})

if (USE_CUDA)
	get_property(GPU_SOURCES_GLOBAL GLOBAL PROPERTY GPU_SOURCES_GLOBAL)
	set_source_files_properties(${GPU_SOURCES_GLOBAL} PROPERTIES LANGUAGE ${locality_aware_LANG})
endif()

### Installation Requirements for Spack Package ###
include(CMakePackageConfigHelpers)

# Install the actual target(s) and register them for export
install(TARGETS locality_aware
        EXPORT locality_awareTargets)

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

