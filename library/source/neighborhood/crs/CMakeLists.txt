file(GLOB neighborhood_HEADERS CONFIGURE_DEPENDS "*.h")
file(GLOB neighborhood_SOURCES CONFIGURE_DEPENDS "*.c" "*.cpp")

set(neighborhood_HEADERS ${neighborhood_HEADERS} CACHE INTERNAL "All headers for neighborhood files.")
set(neighborhood_SOURCES ${neighborhood_SOURCES} CACHE INTERNAL "All source files for neighborhood directory.")

target_sources(locality_aware PRIVATE ${neighborhood_SOURCES})