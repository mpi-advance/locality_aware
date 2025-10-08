file(GLOB collective_HEADERS CONFIGURE_DEPENDS "*.h")
file(GLOB collective_SOURCES CONFIGURE_DEPENDS "*.c")

set(collective_HEADERS ${collective_HEADERS} CACHE INTERNAL "All headers for collective files.")
set(collective_SOURCES ${collective_SOURCES} CACHE INTERNAL "All source files for collective directory.")

set_source_files_properties(${collective_SOURCES} PROPERTIES LANGUAGE CUDA)
message (STATUS "TEST ${collective_SOURCES}")
target_sources(locality_aware PRIVATE ${collective_SOURCES})
