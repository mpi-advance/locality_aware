set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (USE_CUDA)
    set(gpu_util_HEADERS
        src/utils/utils_cuda.h
	)
elseif (USE_HIP)
    set(gpu_util_HEADERS
	    src/utils/utils_hip.h
	)
endif()

file(GLOB utils_SOURCES CONFIGURE_DEPENDS "*.cpp")
set(utils_SOURCES ${utils_SOURCES} PARENT_SCOPE)


set(utils_HEADERS
    src/utils/utils.h
    ${gpu_util_HEADERS}
    PARENT_SCOPE
    )

#set(utils_SOURCES
#    src/utils/utils.cpp
#    PARENT_SCOPE
#    )

