include_directories(".")

add_library(mpi_advance STATIC 
    topology.h
    topology.c
    comm_data.h
    comm_data.c
    comm_pkg.h
    comm_pkg.c
    locality_comm.h
    locality_comm.c
    dist_graph.h
    dist_graph.c
    neighbor.h
    neighbor.c
    )

target_link_libraries(mpi_advance ${MPI_LIBRARIES} stdc++)

