set(CMAKE_INCLUDE_CURRENT_DIR ON)

file(GLOB neighborhood_HEADERS CONFIGURE_DEPENDS "*.h")
file(GLOB neighborhood_SOURCES CONFIGURE_DEPENDS "*.c" "*.cpp")

set(neighborhood_HEADERS ${neighborhood_HEADERS} PARENT_SCOPE)
set(neighborhood_SOURCES ${neighborhood_SOURCES} PARENT_SCOPE)

