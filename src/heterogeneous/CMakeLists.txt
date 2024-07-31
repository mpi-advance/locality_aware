set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(heterogeneous_HEADERS
    heterogeneous/gpu_alltoall.h
    heterogeneous/gpu_alltoallv.h
    heterogeneous/gpu_alltoall_init.h
    PARENT_SCOPE
    )

set(heterogeneous_SOURCES
    heterogeneous/gpu_alltoall.c
    heterogeneous/gpu_alltoallv.c
    heterogeneous/gpu_alltoall_init.c
    PARENT_SCOPE
    )

