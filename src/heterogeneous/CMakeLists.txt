file(GLOB heterogeneous_HEADERS CONFIGURE_DEPENDS "*.h")
file(GLOB heterogeneous_SOURCES CONFIGURE_DEPENDS "*.c")

set(heterogeneous_HEADERS ${heterogeneous_HEADERS} CACHE INTERNAL "All headers for heterogeneous files.")
set(heterogeneous_SOURCES ${heterogeneous_SOURCES} CACHE INTERNAL "All source files for heterogeneous directory.")

message(STATUS ${heterogeneous_HEADERS} ${heterogeneous_SOURCES})
