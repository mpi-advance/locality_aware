target_include_directories(locality_aware PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	$<INSTALL_INTERFACE:include>
)

include_directories(utils)
include_directories(communicator)
include_directories(persistent)
include_directories(collective)
include_directories(neighborhood)
if(USE_CUDA OR USE_HIP)
	include_directories(heterogeneous)
endif()
include_directories(utils)
