target_sources(locality_aware PRIVATE 
	sparse_col/alltoall_crs.cpp
	sparse_col/alltoallv_crs.cpp
	sparse_col/sparse_coll_utils.cpp
	MPIL_Topo/MPIL_Topo_free.cpp
	MPIL_Topo/MPIL_Topo_from_neighbor_comm.cpp
	MPIL_Topo/MPIL_Topo_init.cpp
	locality/neighbor_locality.cpp
	locality/neighbor_alltoallv_locality.cpp
	locality/neighbor_alltoallv_init_locality.cpp
	locality/neighbor_alltoallv_init_locality_ext.cpp
	persistent/init_communication.cpp
	persistent/init_neighbor_request.cpp
	persistent/neighbor_start.cpp
	persistent/neighbor_wait.cpp
	standard/neighbor_alltoallv_standard.cpp
	standard/neighbor_alltoallv_init_standard.cpp
)