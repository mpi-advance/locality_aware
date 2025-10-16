if (USE_CUDA)
		set_source_files_properties(init_request.c PROPERTIES LANGUAGE CUDA)
endif()


target_sources(locality_aware PRIVATE init_request.c)
target_sources(locality_aware PRIVATE allocate_requests.c)