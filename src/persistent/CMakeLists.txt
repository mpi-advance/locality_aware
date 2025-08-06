set(CMAKE_INCLUDE_CURRENT_DIR ON)

file(GLOB persistent_HEADERS CONFIGURE_DEPENDS "*.h")
file(GLOB persistent_SOURCES CONFIGURE_DEPENDS "*.c")

set(persistent_HEADERS ${persistent_HEADERS} CACHE INTERNAL "All headers for persistent files.")
set(persistent_SOURCES ${persistent_SOURCES} CACHE INTERNAL "All source files for persistent directory.")

