set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (USE_CUDA)
    set(gpu_util_HEADERS
        utils/utils_cuda.h
	)
	set(gpu_util_SOURCES
		utils/utils_cuda.c
	)
elseif (USE_HIP)
    set(gpu_util_HEADERS
	    utils/utils_hip.h
	)
	set(gpu_util_SOURCES
		utils/utils_hip.c
	)
endif()



set(utils_HEADERS
    utils/utils.h
	${gpu_util_HEADERS}
    PARENT_SCOPE
    )

set(utils_SOURCES
    utils/utils.cpp
	${gpu_util_SOURCES}
    PARENT_SCOPE
    )

