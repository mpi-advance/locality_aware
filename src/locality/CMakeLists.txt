set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(locality_HEADERS
    locality/comm_data.h
    locality/comm_pkg.h
    locality/locality_comm.h
    locality/topology.h
    PARENT_SCOPE
    )

set(locality_SOURCES
    locality/comm_data.c
    locality/comm_pkg.c
    locality/locality_comm.c
    locality/topology.c
    PARENT_SCOPE
    )

