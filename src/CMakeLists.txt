include_directories(".")

add_library(mpi_advance STATIC 
    nap_comm.h
    nap_comm.c
    nap_communicate.cpp
    topo_cray.hpp
    topo_data.hpp
    topo_default.hpp
    )

target_link_libraries(mpi_advance ${MPI_LIBRARIES} )

