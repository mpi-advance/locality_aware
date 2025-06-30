set(CMAKE_INCLUDE_CURRENT_DIR ON)

file(GLOB heterogeneous_HEADERS CONFIGURE_DEPENDS "*.h")
file(GLOB heterogeneous_SOURCES CONFIGURE_DEPENDS "*.c")

set(heterogeneous_HEADERS ${heterogeneous_HEADERS} PARENT_SCOPE)
set(heterogeneous_SOURCES ${heterogeneous_SOURCES} PARENT_SCOPE)

message(STATUS ${heterogeneous_HEADERS} ${heterogeneous_SOURCES})
