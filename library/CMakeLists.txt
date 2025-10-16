target_include_directories(
    locality_aware PUBLIC $<BUILD_INTERFACE : ${CMAKE_CURRENT_SOURCE_DIR} / include>)
    add_subdirectory(bindings) add_subdirectory(source) add_subdirectory(tests)
