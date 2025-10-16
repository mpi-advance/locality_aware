file(GLOB communicator_bindings CONFIGURE_DEPENDS "*.c" "*.cpp")
set(communicator_SOURCES ${communicator_bindings} CACHE INTERNAL "All source files for communicator directory.")
target_sources(locality_aware PRIVATE ${communicator_bindings})

