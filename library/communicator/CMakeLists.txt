file(GLOB communicator_HEADERS CONFIGURE_DEPENDS "*.h")
file(GLOB communicator_SOURCES CONFIGURE_DEPENDS "*.c")

set(communicator_HEADERS ${communicator_HEADERS} CACHE INTERNAL "All headers for communicator files.")
set(communicator_SOURCES ${communicator_SOURCES} CACHE INTERNAL "All source files for communicator directory.")

target_sources(locality_aware PRIVATE ${communicator_SOURCES})