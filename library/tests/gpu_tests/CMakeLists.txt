## Grab all .cpp files in test directory
file(GLOB CPP_SOURCES CONFIGURE_DEPENDS "*.cpp")

## All should be compiled appropriately (cxx vs nvcc vs hipcc)
set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE ${locality_aware_LANG})

## Go through each CPP file
foreach(src ${CPP_SOURCES})
    make_test(${src})
endforeach()


