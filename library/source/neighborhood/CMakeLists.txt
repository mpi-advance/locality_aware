target_sources(locality_aware PRIVATE 
	sparse_col/alltoall_crs.cpp
	sparse_col/alltoallv_crs.cpp
	sparse_col/sparse_coll_utils.cpp
	MPIL_Topo/MPIL_Topo_free.c
	MPIL_Topo/MPIL_Topo_from_neighbor_comm.c
	MPIL_Topo/MPIL_Topo_init.c
	locality/neighbor_locality.cpp
	locality/neighbor_alltoallv_locality.c
	locality/neighbor_alltoallv_init_locality.c
	locality/neighbor_alltoallv_init_locality_ext.c
	persistent/init_communication.c
	persistent/init_neighbor_request.c
	persistent/neighbor_start.c
	persistent/neighbor_wait.c
	standard/neighbor_alltoallv_standard.c
	standard/neighbor_alltoallv_init_standard.c
)