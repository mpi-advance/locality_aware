set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(neighborhood_HEADERS
    neighborhood/dist_graph.h
    neighborhood/neighbor.h
    neighborhood/neighbor_persistent.h
    PARENT_SCOPE
    )

set(neighborhood_SOURCES
    neighborhood/dist_graph.c
    neighborhood/neighbor.c
    neighborhood/neighbor_locality.cpp
    neighborhood/neighbor_persistent.c
    PARENT_SCOPE
    )

