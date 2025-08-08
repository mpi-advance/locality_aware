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
set(utils_SOURCES ${utils_SOURCES} CACHE INTERNAL "All source files for utils directory.")


set(utils_HEADERS
    src/utils/utils.h
    ${gpu_util_HEADERS}
    CACHE INTERNAL "All headers for utils files."
    )

