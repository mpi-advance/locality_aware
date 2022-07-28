set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(collective_HEADERS
    collective/collective.h
    collective/allgather.h
    PARENT_SCOPE
    )

set(collective_SOURCES
    collective/alltoall.c
    collective/alltoallv.c
    collective/allgather.c
    PARENT_SCOPE
    )

