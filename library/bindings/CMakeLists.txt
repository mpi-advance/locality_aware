target_sources(locality_aware PRIVATE global_defaults.c)

target_sources(locality_aware PRIVATE Collective/MPIL_Alltoall.c)
target_sources(locality_aware PRIVATE Collective/MPIL_Alltoallv.c)

target_sources(locality_aware PRIVATE CRS/MPIL_Alltoall_crs.c)
target_sources(locality_aware PRIVATE CRS/MPIL_Alltoallv_crs.c)

target_sources(locality_aware PRIVATE MPIL_Alloc/MPIL_Alloc.cpp)
target_sources(locality_aware PRIVATE MPIL_Alloc/MPIL_Free.cpp)

target_sources(locality_aware PRIVATE MPIL_Info/MPIL_Info_init.cpp)
target_sources(locality_aware PRIVATE MPIL_Info/MPIL_Info_free.cpp)

target_sources(locality_aware PRIVATE MPIL_Request/MPIL_Start.c)
target_sources(locality_aware PRIVATE MPIL_Request/MPIL_Wait.c)
target_sources(locality_aware PRIVATE MPIL_Request/MPIL_Request_free.c)

add_subdirectory(Neighborhood)
add_subdirectory(MPIL_Comm)


