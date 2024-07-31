set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(collective_HEADERS
    collective/collective.h
    collective/alltoall.h
    collective/alltoallv.h
    collective/alltoall_init.h
    PARENT_SCOPE
    )

set(collective_SOURCES
    collective/alltoall.c
    collective/alltoallv.c
    collective/alltoall_init.c
    PARENT_SCOPE
    )

