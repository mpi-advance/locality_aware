set(CMAKE_INCLUDE_CURRENT_DIR ON)

file(GLOB communicator_HEADERS CONFIGURE_DEPENDS "*.h")
file(GLOB communicator_SOURCES CONFIGURE_DEPENDS "*.c")

set(communicator_HEADERS ${communicator_HEADERS} PARENT_SCOPE)
set(communicator_SOURCES ${communicator_SOURCES} PARENT_SCOPE)

