set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(collective_HEADERS
    collective/collective.h
    collective/alltoall.h
    collective/alltoallv.h
    collective/allgather.h
    collective/gather.h
    collective/bcast.h
    PARENT_SCOPE
    )

set(collective_SOURCES
    collective/alltoall.c
    collective/alltoallv.c
    collective/allgather.c
    collective/gather.c
    collective/bcast.c
    PARENT_SCOPE
    )

