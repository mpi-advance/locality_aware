set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(neighborhood_HEADERS
    neighborhood/dist_topo.h
    neighborhood/dist_graph.h
    neighborhood/dist_topo.h
    neighborhood/neighbor.h
    neighborhood/sparse_coll.h
    PARENT_SCOPE
    )

set(neighborhood_SOURCES
    neighborhood/dist_topo.c
    neighborhood/dist_graph.c
    neighborhood/dist_topo.c
    neighborhood/neighbor.c
    neighborhood/neighbor_locality.cpp
    neighborhood/sparse_coll.c
    neighborhood/sparse_coll_utils.cpp
    PARENT_SCOPE
    )

