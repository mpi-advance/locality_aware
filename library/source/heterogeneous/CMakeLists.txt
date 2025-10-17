file(GLOB heterogeneous_SOURCES CONFIGURE_DEPENDS "*.c" "*.cpp")
set(heterogeneous_SOURCES ${heterogeneous_SOURCES} CACHE INTERNAL "All source files for heterogeneous directory.")

#always add to gpu language converision list. 
get_property(GPU_SOURCES GLOBAL PROPERTY GPU_SOURCES_GLOBAL)
foreach(file ${heterogeneous_SOURCES})	
	set_property(GLOBAL APPEND PROPERTY GPU_SOURCES_GLOBAL  "${file}")
endforeach()

target_sources(locality_aware PRIVATE ${heterogeneous_SOURCES})
