set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(neighborhood_HEADERS
    neighborhood/dist_topo.h
    neighborhood/dist_graph.h
    neighborhood/neighbor.h
    PARENT_SCOPE
    )

set(neighborhood_SOURCES
    neighborhood/dist_topo.c
    neighborhood/dist_graph.c
    neighborhood/neighbor.c
    neighborhood/neighbor_locality.cpp
    PARENT_SCOPE
    )

