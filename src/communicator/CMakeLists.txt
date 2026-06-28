set(CMAKE_INCLUDE_CURRENT_DIR ON)

file(GLOB communicator_HEADERS CONFIGURE_DEPENDS "*.h")
file(GLOB communicator_SOURCES CONFIGURE_DEPENDS "*.c")

set_source_files_properties(
  ${communicator_HEADERS} ${commmunicator_SOURCES}
 PROPERTIES LANGUAGE ${locality_aware_LANG})


set(communicator_HEADERS ${communicator_HEADERS} PARENT_SCOPE)
set(communicator_SOURCES ${communicator_SOURCES} PARENT_SCOPE)

