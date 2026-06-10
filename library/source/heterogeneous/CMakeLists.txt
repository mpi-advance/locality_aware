#always add to gpu language conversion list. 
get_property(GPU_SOURCES GLOBAL PROPERTY GPU_SOURCES_GLOBAL)
set_property(GLOBAL APPEND PROPERTY GPU_SOURCES_GLOBAL  
	${CMAKE_CURRENT_SOURCE_DIR}/gpu_check.c
)

target_sources(locality_aware PRIVATE 
	gpu_check.c
)
