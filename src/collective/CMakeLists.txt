set(CMAKE_INCLUDE_CURRENT_DIR ON)
file(GLOB collective_HEADERS CONFIGURE_DEPENDS "*.h")
file(GLOB collective_SOURCES CONFIGURE_DEPENDS "*.c")

set_source_files_properties(
  ${collective_HEADERS} ${collective_SOURCES}
 PROPERTIES LANGUAGE ${locality_aware_LANG})

set(collective_HEADERS ${collective_HEADERS} CACHE INTERNAL "All headers for collective files.")
set(collective_SOURCES ${collective_SOURCES} CACHE INTERNAL "All source files for collective directory.")

