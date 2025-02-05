# search prefix path
set(MPIPCL_PREFIX "${CMAKE_PREFIX_PATH}" CACHE STRING "Help cmake to find MPIPCL")

# check include
find_path(MPIPCL_INCLUDE_DIR NAMES mpipcl.h HINTS ${MPIPCL_PREFIX}/include)

# check lib
find_library(MPIPCL_LIBRARY NAMES mpipcl
	HINTS ${MPIPCL_PREFIX}/lib)

# setup found
if (MPIPCL_INCLUDE_DIR AND MPIPCL_LIBRARY)
	set(MPIPCL_FOUND ON)
endif()

# handle QUIET/REQUIRED
include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set MPIPCL_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(MPIPCL DEFAULT_MSG MPIPCL_INCLUDE_DIR MPIPCL_LIBRARY)

# Hide internal variables
mark_as_advanced(MPIPCL_INCLUDE_DIR MPIPCL_FOUND MPIPCL_LIBRARY MPIPCL_PREFIX)