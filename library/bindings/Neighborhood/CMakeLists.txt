file(GLOB neighborhood_bindings CONFIGURE_DEPENDS "*.c" "*.cpp")
set(communicator_SOURCES ${neighborhood_bindings} CACHE INTERNAL "All source files for communicator directory.")
target_sources(locality_aware PRIVATE ${neighborhood_bindings})