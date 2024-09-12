set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (USE_CUDA)
    set(gpu_util_HEADERS
        utils/utils_cuda.h
	)
elseif (USE_HIP)
    set(gpu_util_HEADERS
	    utils/utils_hip.h
	)
endif()



set(utils_HEADERS
    utils/utils.h
	${gpu_util_HEADERS}
    PARENT_SCOPE
    )

set(utils_SOURCES
    utils/utils.cpp
    PARENT_SCOPE
    )

